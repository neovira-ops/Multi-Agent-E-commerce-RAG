"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline (uses the AgentCore CLI)
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)

Deployment uses the AgentCore CLI (`agentcore`, npm package @aws/agentcore,
https://github.com/aws/agentcore-cli) through the pre-written helper
src/agentcore_cli.py - see the README for the prerequisites (Node.js 20+, uv).
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    model = BedrockModel(
    model_id=config.WORKER_MODEL_ID,
    region_name=config.AWS_REGION,
    temperature=0.1,
)

    system_prompt = """
You are InventoryAgent, a specialist data retrieval agent
for NovaMart customer support.

Your responsibility is to retrieve accurate customer and
order information from DynamoDB using your available tools.

You can:
- Check the status and details of a specific order.
- Retrieve a customer's account profile and membership tier.
- List all orders belonging to a customer.

Rules:
1. Always use the appropriate tool to retrieve factual data.
2. Never invent customer details, order information, or statuses.
3. Never make refund or return eligibility decisions.
4. Never modify customer or order records.
5. Report missing records clearly.
6. Return accurate, structured findings for the Orchestrator
   and downstream agents.

You are a data-gathering agent, not a decision-making agent.
"""

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Retrieve a specific customer's order from DynamoDB.

        Args:
            customer_id: The customer's unique identifier.
            order_id: The order's unique identifier.

        Returns:
            Order details, or an error if the order cannot be found.
        """
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)

            response = table.get_item(
                Key={
                    "customer_id": customer_id,
                    "order_id": order_id,
            }
        )

            order = response.get("Item")

            if not order:
                return {
                    "success": False,
                    "message": "Order not found",
                    "customer_id": customer_id,
                    "order_id": order_id,
            }

            return {
                "success": True,
                "order": order,
        }

        except Exception as e:
            logger.exception("Failed to retrieve order")
            return {
                "success": False,
                "error": str(e),
            }

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        try:
            table = dynamodb.Table(config.CUSTOMERS_TABLE)
            response = table.get_item(Key={"customer_id": customer_id})
            customer = response.get("Item")

            if not customer:
                return {
                    "success": False,
                    "message": "Customer not found",
                    "customer_id": customer_id,
                }

            return {
                "success": True,
                "tier": customer.get("tier"),
                "customer": customer,
            }

        except Exception as e:
            logger.exception("Failed to retrieve customer tier")
            return {
                "success": False,
                "error": str(e),
            }

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)

            response = table.query(
                KeyConditionExpression=Key("customer_id").eq(customer_id)
            )

            orders = response.get("Items", [])

            while "LastEvaluatedKey" in response:
                response = table.query(
                    KeyConditionExpression=Key("customer_id").eq(customer_id),
                    ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                orders.extend(response.get("Items", []))

            return {
                "success": True,
                "customer_id": customer_id,
                "order_count": len(orders),
                "orders": orders,
            }

        except Exception as e:
            logger.exception("Failed to retrieve customer orders")
            return {
                "success": False,
                "error": str(e),
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            check_order_status,
            get_customer_tier,
            list_customer_orders,
        ],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    system_prompt = """
You are RefundAgent, the returns and refunds specialist
for NovaMart customer support.

Your responsibility is to evaluate return eligibility using
verified order and customer information from WorkflowState.

Follow these rules:

1. Always call get_inventory_context before making a decision.
2. Use the order date and customer membership tier to determine
   whether the order falls within the permitted return window.
3. Standard customers have a 30-day return window.
4. Premium customers have a 60-day return window.
5. Never invent order dates, customer tiers, or order statuses.
6. If required information is missing, explain what is missing.
7. Only call initiate_refund when the return is eligible and
   the customer has requested the return.
8. Never initiate a refund for an ineligible order.
9. Report your decision and supporting facts clearly so the
   Orchestrator can store the result in WorkflowState.

You are responsible for refund eligibility decisions.
You do not compose the final customer-facing response.
"""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        try:
                    state = _read_workflow_state(session_id)
        
                    if not state:
                        return {}
        
                    return state.get("inventory_agent") or {}
        
        except Exception:
                    logger.exception("Failed to retrieve inventory context")
                    return {}

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        from datetime import datetime, timezone
        from botocore.exceptions import ClientError

        try:
            orders_table = dynamodb.Table(config.ORDERS_TABLE)
            customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)

            customer_response = customers_table.get_item(
                Key={"customer_id": customer_id}
            )
            customer = customer_response.get("Item")

            if not customer:
                return {
                    "success": False,
                    "message": "Customer not found",
                }

            tier = customer.get("tier", "Standard")
            return_window = 60 if tier.lower() == "premium" else 30

            order_response = orders_table.get_item(
                Key={
                    "customer_id": customer_id,
                    "order_id": order_id,
                }
            )
            order = order_response.get("Item")

            if not order:
                return {
                    "success": False,
                    "message": "Order not found",
                }

            if order.get("status") != "delivered":
                return {
                    "success": False,
                    "message": (
                        "Order cannot be returned because "
                        "its status is not delivered."
                    ),
                }

            order_date = datetime.strptime(
                order["order_date"], "%Y-%m-%d"
            ).date()

            today = datetime.now(timezone.utc).date()
            days_since_order = (today - order_date).days

            if not 0 <= days_since_order <= return_window:
                return {
                    "success": False,
                    "message": "Order is outside the return window.",
                    "customer_tier": tier,
                    "return_window_days": return_window,
                    "days_since_order": days_since_order,
                }

            return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"

            orders_table.update_item(
                Key={
                    "customer_id": customer_id,
                    "order_id": order_id,
                },
                UpdateExpression=(
                    "SET #status = :new_status, "
                    "return_reference = :reference, "
                    "return_reason = :reason"
                ),
                ConditionExpression="#status = :delivered",
                ExpressionAttributeNames={
                    "#status": "status",
                },
                ExpressionAttributeValues={
                    ":new_status": "return_requested",
                    ":reference": return_reference,
                    ":reason": reason,
                    ":delivered": "delivered",
                },
            )

            return {
                "success": True,
                "customer_id": customer_id,
                "order_id": order_id,
                "return_reference": return_reference,
                "status": "return_requested",
                "instructions": (
                    "Your return request has been recorded. "
                    "Keep the item and its packaging available "
                    "for further return instructions."
                ),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == (
                "ConditionalCheckFailedException"
            ):
                return {
                    "success": False,
                    "message": (
                        "The order status has changed or "
                        "a return has already been requested."
                    ),
                }

            logger.exception("DynamoDB refund operation failed")
            return {
                "success": False,
                "message": "Unable to process the return request.",
            }

        except (ValueError, KeyError):
            logger.exception("Invalid order data")
            return {
                "success": False,
                "message": "Order data is incomplete or invalid.",
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            get_inventory_context,
            initiate_refund,
        ],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        results = retrieve_from_knowledge_base(
        kb_id=config.RETURNS_KB_ID,
        query=query,
        top_k=3,
        )

        if not results:
            return "No Returns Policy information was retrieved."

        return format_kb_results(results)

    # Create the ReturnsPolicyRetrieverAgent with the tool above
    returns_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt="""
You are NovaMart's Returns Policy Retriever.

Your responsibility is to retrieve factual information
from the Returns Policy Knowledge Base.

Rules:
1. Always call retrieve_returns_policy for the question.
2. Use only the retrieved policy passages.
3. Do not invent return policies or eligibility rules.
4. If no relevant information is retrieved, clearly
   state that the information is unavailable.
5. Return the relevant policy findings to PolicyAgent.
6. Do not compose the final customer-facing response.
""",
        tools=[retrieve_returns_policy],
    )

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        results = retrieve_from_knowledge_base(
        kb_id=config.SHIPPING_KB_ID,
        query=query,
        top_k=3,
        )

        if not results:
            return "No Shipping Policy information was retrieved."

        return format_kb_results(results)

    # Create the ShippingPolicyRetrieverAgent with the tool above
    shipping_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt="""
You are NovaMart's Shipping Policy Retriever.

Your responsibility is to retrieve accurate information
from the Shipping Policy Knowledge Base.

Rules:
1. Always call retrieve_shipping_policy for the question.
2. Use only information found in retrieved policy passages.
3. Do not invent shipping rates, delivery times, or policies.
4. If no relevant information is retrieved, clearly
   state that the information is unavailable.
5. Return relevant findings to PolicyAgent.
6. Do not compose the final customer-facing response.
""",
        tools=[retrieve_shipping_policy],
    )

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        results = retrieve_from_knowledge_base(
        kb_id=config.WARRANTY_KB_ID,
        query=query,
        top_k=3,
        )

        if not results:
            return "No Warranty Policy information was retrieved."

        return format_kb_results(results)

    # Create the WarrantyPolicyRetrieverAgent with the tool above
    warranty_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt="""
You are NovaMart's Warranty Policy Retriever.

Your responsibility is to retrieve accurate information
from the Warranty Policy Knowledge Base.

Rules:
1. Always call retrieve_warranty_policy for the question.
2. Use only information from retrieved policy passages.
3. Do not invent warranty periods, coverage, or exclusions.
4. If no relevant information is retrieved, clearly
   state that the information is unavailable.
5. Return relevant findings to PolicyAgent.
6. Do not compose the final customer-facing response.
""",
        tools=[retrieve_warranty_policy],
    )

    _apply_guardrail([returns_retriever, shipping_retriever, warranty_retriever])

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        # Build a dict mapping domain names to their retriever agents
        # e.g. {'Returns': returns_retriever, 'Shipping': shipping_retriever, ...}
        retrievers = {
            "Returns": returns_retriever,
            "Shipping": shipping_retriever,
            "Warranty": warranty_retriever,
        }
        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        # Define a helper to run one retriever sub-agent
        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            This covers both the direct worker thread and any internal
            streaming child threads that Strands SDK spawns internally -
            which do NOT inherit thread-local variables and therefore cannot
            be suppressed with a thread-local capture approach.
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            try:
                response = agent(query)
                return domain, str(response)
            except Exception:
                logger.exception(
                    "Policy retriever failed: %s", domain
                )
                return domain, "[Retrieval failed]"

        # Use ThreadPoolExecutor to run all three retrievers in parallel
        # Collect results into a dict: {'Returns': '...', 'Shipping': '...', ...}
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    _run_retriever, domain, agent, query
                ): domain
                for domain, agent in retrievers.items()
            }

            for future in as_completed(futures):
                domain, result_text = future.result()
                results[domain] = result_text

        # ── Trace: all KBs responded - print each result sequentially ─────
        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        # Combine results from all three domains and return
        combined = []

        for domain in ["Returns", "Shipping", "Warranty"]:
            combined.append(
                f"=== {domain} Policy ===\n"
                f"{results.get(domain, '[No results]')}"
            )

        return "\n\n".join(combined)

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2,
    )

    system_prompt = """
You are PolicyAgent, NovaMart's policy research coordinator.

You coordinate three specialist policy retriever agents:
ReturnsPolicyRetriever, ShippingPolicyRetriever, and
WarrantyPolicyRetriever.

Your responsibilities:

1. Always call search_all_policies before answering a policy question.
2. The tool runs all three specialist retrievers in parallel.
3. Review the retrieved passages from all three policy domains.
4. Answer using only information supported by retrieved documents.
5. Do not invent policy rules, return windows, shipping rates,
   warranty coverage, or exclusions.
6. If the retrieved information is insufficient, clearly state
   what could not be verified.
7. Distinguish between Returns, Shipping, and Warranty policies.
8. Do not retrieve customer account details or order records.
   Those responsibilities belong to InventoryAgent.
9. Return your grounded policy findings to OrchestratorAgent.
10. Do not compose the final customer-facing response.

Your role is policy research and synthesis, not customer communication.
"""

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    model = BedrockModel(
    model_id=config.WORKER_MODEL_ID,
    region_name=config.AWS_REGION,
    temperature=0.3,
    )

    system_prompt = """
You are CommunicationAgent, NovaMart's customer communication specialist.

Your responsibility is to prepare the final customer-facing response
using findings from the shared WorkflowState.

Rules:

1. Always call get_full_workflow_context before composing a response.
2. Use the InventoryAgent findings for customer, order, and delivery facts.
3. Use the PolicyAgent findings for returns, shipping, and warranty rules.
4. Use the RefundAgent findings for return eligibility and refund actions.
5. Never claim a refund or return was initiated unless the workflow
   confirms that the action succeeded.
6. Never invent order statuses, refund references, policy details,
   delivery dates, or customer information.
7. If information is unavailable, explain that clearly.
8. Write in a professional, friendly, empathetic tone.
9. Keep responses concise, clear, and actionable.
10. Include relevant next steps when appropriate.
11. Do not initiate refunds, modify orders, or change customer records.
12. Do not expose internal agent names, tool calls, or WorkflowState
    implementation details to customers.

Your output is the final customer-facing message.
"""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        try:
            state = _read_workflow_state(session_id)

            if not state:
                return {
                    "success": False,
                    "message": "No workflow state found for this session.",
                    "session_id": session_id,
                }

            return {
                "success": True,
                "session_id": session_id,
                "inventory_agent": state.get("inventory_agent", {}),
                "policy_agent": state.get("policy_agent", {}),
                "refund_agent": state.get("refund_agent", {}),
            }

        except Exception as e:
            logger.exception(
                "Failed to retrieve workflow context for session %s",
                session_id,
            )

            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    model = BedrockModel(
    model_id=config.ORCHESTRATOR_MODEL_ID,
    region_name=config.AWS_REGION,
    temperature=0.0,
    )

    system_prompt = """
You are OrchestratorAgent, the central workflow coordinator
for NovaMart's multi-agent customer support system.

You coordinate four specialist agents:
- InventoryAgent: Customer accounts, order status, and order history.
- PolicyAgent: Returns, shipping, and warranty policy research.
- RefundAgent: Return eligibility and refund processing.
- CommunicationAgent: Final customer-facing responses.

WORKFLOW RULES:

1. ALWAYS call initialize_session at the beginning of every
   customer request, before routing to any worker agent.

2. For order status, order history, returns, or refunds:
   Call route_to_inventory_agent first to gather order facts.
   For return or refund processing, call route_to_refund_agent
   after InventoryAgent.

3. For questions about return policies, shipping policies,
   or warranties, call route_to_policy_agent.

4. For customer account details or membership tiers,
   use InventoryAgent, never PolicyAgent.

5. For requests involving only calculations or arithmetic,
   route directly to CommunicationAgent after initialization.
   Do not invoke InventoryAgent, PolicyAgent, or RefundAgent
   unless additional information is required.

6. ALWAYS call route_to_communication_agent LAST, after
   all relevant specialist agents have completed their work.

ADDITIONAL RULES:

- Use the session_id and customer_id provided in the request.
- Preserve the customer's original request when routing.
- Do not invent customer information, order details,
  policy rules, or refund outcomes.
- Never initiate a refund unless the customer requested
  the return or refund action.
- Do not bypass InventoryAgent for refund requests.
- Do not answer the customer directly.
- Return the final response produced by CommunicationAgent.
- Do not expose internal workflow details to customers.
"""

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        try:
            state = _read_workflow_state(session_id)

            if not state:
                return "Inventory routing failed: session not initialized."

            if state.get("customer_id") != customer_id:
                return "Inventory routing failed: customer does not match session."

            agent_request = (
                f"Customer ID: {customer_id}\n"
                f"Customer request: {request}\n\n"
                "Retrieve the relevant customer and order facts "
                "using your available tools. "
                "Do not make refund eligibility decisions."
            )

            response = inventory_agent(agent_request)
            result_text = str(response)

            _update_workflow_state(
                session_id=session_id,
                updates={
                    "inventory_agent": result_text,
                },
                expected_version=int(state["version"]),
            )

            return result_text

        except Exception:
            logger.exception(
                "InventoryAgent routing failed for session %s",
                session_id,
            )

            return "InventoryAgent could not complete the request."

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        try:
            state = _read_workflow_state(session_id)

            if not state:
                return "Policy routing failed: session not initialized."

            agent_request = (
                f"Customer policy question: {request}\n\n"
                "Search the Returns, Shipping, and Warranty "
                "knowledge bases using search_all_policies. "
                "Summarize only information supported by "
                "the retrieved policy documents. "
                "Clearly identify any information that "
                "could not be verified."
            )

            response = policy_agent(agent_request)
            result_text = str(response)

            _update_workflow_state(
                session_id=session_id,
                updates={
                    "policy_agent": result_text,
                },
                expected_version=int(state["version"]),
            )

            return result_text

        except Exception:
            logger.exception(
                "PolicyAgent routing failed for session %s",
                session_id,
            )

            return "PolicyAgent could not complete the request."

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        try:
            state = _read_workflow_state(session_id)

            if not state:
                return "Refund routing failed: session not initialized."

            if state.get("customer_id") != customer_id:
                return "Refund routing failed: customer does not match session."

            inventory_context = state.get("inventory_agent")

            if not inventory_context:
                return (
                    "Refund routing failed: InventoryAgent must "
                    "run before RefundAgent."
                )

            agent_request = (
                f"Session ID: {session_id}\n"
                f"Customer ID: {customer_id}\n"
                f"Customer request: {request}\n\n"
                "First call get_inventory_context using the session ID. "
                "Review the order facts and return eligibility. "
                "Only initiate a return or refund if the customer "
                "explicitly requested that action and all required "
                "conditions are satisfied. "
                "Do not invent order information."
            )

            response = refund_agent(agent_request)
            result_text = str(response)

            _update_workflow_state(
                session_id=session_id,
                updates={"refund_agent": result_text},
                expected_version=int(state["version"]),
            )

            return result_text

        except Exception:
            logger.exception(
                "RefundAgent routing failed for session %s",
                session_id,
            )
            return "RefundAgent could not complete the request."

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        try:
            state = _read_workflow_state(session_id)

            if not state:
                return "Communication routing failed: session not initialized."

            if state.get("customer_id") != customer_id:
                return "Communication routing failed: customer does not match session."

            agent_request = (
                f"Session ID: {session_id}\n"
                f"Customer ID: {customer_id}\n"
                f"Original customer request: {original_request}\n\n"
                "First call get_full_workflow_context using the session ID. "
                "Review the findings from InventoryAgent, PolicyAgent, "
                "and RefundAgent that are relevant to this request. "
                "Compose a clear, professional, empathetic customer response. "
                "Only describe actions as completed when the workflow "
                "confirms they succeeded. "
                "Do not invent missing information or expose internal details."
            )

            response = communication_agent(agent_request)
            result_text = str(response)

            _update_workflow_state(
                session_id=session_id,
                updates={
                    "communication_agent": result_text,
                },
                expected_version=int(state["version"]),
            )

            return result_text

        except Exception:
            logger.exception(
                "CommunicationAgent routing failed for session %s",
                session_id,
            )

            return "CommunicationAgent could not complete the request."

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            state = _create_workflow_state(
                session_id=session_id,
                customer_id=customer_id,
            )

            return (
                f"Session initialized successfully. "
                f"Session ID: {state['session_id']}. "
                f"Customer ID: {state['customer_id']}."
            )

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            existing_state = _read_workflow_state(session_id)

            if existing_state and existing_state.get("customer_id") == customer_id:
                return (
                    f"Session already exists for customer {customer_id}. "
                    f"Continuing with session {session_id}."
                )

            return (
                "Session initialization failed: "
                "the session ID is already assigned to another customer."
            )

        except Exception:
            logger.exception(
                "Failed to initialize workflow session %s",
                session_id,
            )

            return "Session initialization failed due to an internal error."

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT TOOLING - AgentCore CLI
#
#  The runtime is deployed with the AgentCore CLI (`agentcore`, npm package
#  @aws/agentcore - https://github.com/aws/agentcore-cli) through the
#  pre-written helper src/agentcore_cli.py:
#
#    agentcore_cli.stage_runtime_code()   copies this file, its helper modules
#                                         and config.py to build/runtime/ with a
#                                         pyproject.toml of the runtime deps
#    agentcore_cli.configure_runtime()    writes the runtime settings (network
#                                         mode, protocol, execution role, env
#                                         vars) to agentcore/agentcore.json
#    agentcore_cli.deploy()               runs `agentcore deploy -y`: the CLI
#                                         downloads arm64 / Python 3.12 wheels
#                                         with uv, zips them with the code
#                                         (direct code deployment) and creates
#                                         or updates the runtime via CDK
#    agentcore_cli.deployed_runtime_arn() reads the ARN the CLI recorded
#
#  Inside the runtime this same file is the entry point: it is started with
#  no command-line argument and serves HTTP (see run_serve). The marker file
#  written next to it by stage_runtime_code() tells __main__ to do so.
# ═══════════════════════════════════════════════════════

_RUNTIME_MARKER = '.agentcore-runtime'         # written by agentcore_cli.stage_runtime_code()
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    # Reuse an existing guardrail without creating duplicates.
    existing = bedrock_client.list_guardrails()

    for g in existing.get("guardrails", []):
        if g["name"] != config.GUARDRAIL_NAME:
            continue

        guardrail_id = g["id"]

        versions = bedrock_client.list_guardrails(
            guardrailIdentifier=guardrail_id,
        )

        numbered_versions = [
            v["version"]
            for v in versions.get("guardrails", [])
            if v.get("version", "DRAFT").isdigit()
        ]

        if numbered_versions:
            guardrail_version = str(
                max(int(v) for v in numbered_versions)
            )
        else:
            # Publish DRAFT rather than returning it.
            version_response = bedrock_client.create_guardrail_version(
                guardrailIdentifier=guardrail_id,
                description="NovaMart published guardrail version",
            )
            guardrail_version = version_response["version"]

        print(
            f"Guardrail already exists: {guardrail_id} "
            f"(version: {guardrail_version})"
        )

        return guardrail_id, guardrail_version

    # Use bedrock_client.create_guardrail() with:
    #   - Content policy - block harmful categories at HIGH strength
    #   - PII policy - block credit cards + SSNs; anonymize emails + phone numbers
    #   - Topic policy - deny off-topic subjects (competitor_products, legal_threats, pricing_negotiations)
    #   - Word policy - profanity filter
    #   - blockedInputMessaging and blockedOutputsMessaging

    # Promote from DRAFT to a versioned guardrail using create_guardrail_version()

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description="NovaMart customer support safety guardrail",
        blockedInputMessaging=(
            "I'm sorry, but I can't help with that request. "
            "Please ask about NovaMart orders, products, or policies."
        ),
        blockedOutputsMessaging=(
            "I'm sorry, but I can't provide that response. "
            "Please rephrase your question."
        ),
        contentPolicyConfig={
            "filtersConfig": [
                {
                    "type": category,
                    "inputStrength": strength,
                    "outputStrength": strength,
                }
                for category, strength in [
                    ("SEXUAL", "HIGH"),
                    ("VIOLENCE", "HIGH"),
                    ("HATE", "HIGH"),
                    ("INSULTS", "MEDIUM"),
                    ("MISCONDUCT", "MEDIUM"),
                ]
            ]
        },
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
                {"type": "EMAIL", "action": "ANONYMIZE"},
                {"type": "PHONE", "action": "ANONYMIZE"},
            ]
        },
        topicPolicyConfig={
            "tierConfig": {"tierName": "STANDARD"},
            "topicsConfig": [
                {
                    "name": "competitor_products",
                    "definition": (
                        "Requests to compare, recommend, or promote "
                        "competitor products instead of NovaMart products."
                    ),
                    "examples": [
                        "Which competitor should I buy from instead?",
                        "Recommend products from another retailer.",
                    ],
                    "type": "DENY",
                },
                {
                    "name": "pricing_negotiations",
                    "definition": (
                        "Requests to haggle, negotiate, or change an "
                        "advertised product price. Excludes mathematical "
                        "calculations using a specified price and discount."
                    ),
                    "examples": [
                        "Can you lower the advertised price?",
                        "I'll buy it if you give me a better price.",
                    ],
                    "type": "DENY",
                },
                {
                    "name": "legal_threats",
                    "definition": (
                        "Threats of lawsuits or legal action against "
                        "NovaMart intended to pressure customer support."
                    ),
                    "examples": [
                        "I'll sue NovaMart unless you refund me.",
                        "My lawyer will take legal action against you.",
                    ],
                    "type": "DENY",
                },
            ],
        },
        wordPolicyConfig={
            "managedWordListsConfig": [
                {"type": "PROFANITY"}
            ]
        },
        crossRegionConfig={
            "guardrailProfileIdentifier": "us.guardrail.v1:0"
        },
    )

    guardrail_id = response["guardrailId"]

    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="NovaMart production guardrail version",
    )

    guardrail_version = version_response["version"]

    print(
        f"Guardrail created: {guardrail_id} "
        f"(version: {guardrail_version})"
    )

    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime with
    the AgentCore CLI (src/agentcore_cli.py wraps it).

    AgentCore does not serialize Python objects, so `orchestrator_agent` is
    not uploaded directly. Instead the pre-written staging step copies this
    file, which doubles as the HTTP entry point (see run_serve), together
    with its helper modules and config.py to build/runtime/. `agentcore
    deploy` then packages that directory with arm64 dependencies and creates
    or updates the runtime ("direct code deployment"). Re-running is safe:
    an unchanged runtime is left alone, a changed one is updated in place.

    The guardrail is attached by environment variables: inside the runtime
    build_agent_graph() reads GUARDRAIL_ID / GUARDRAIL_VERSION and applies
    them to every agent's model (see _apply_guardrail), exactly as `test`
    and `chat` do locally.

    Returns:
        The AgentCore Runtime ARN
    """
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME
    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")
    print(f"  Runtime: {runtime_name}  |  CLI project: agentcore/agentcore.json "
          f"(stack {agentcore_cli.stack_name()})")
    previous_arn = agentcore_cli.deployed_runtime_arn()
    if previous_arn:
        print(f"  Runtime already deployed - updating it: {previous_arn}")

    # Stage the code the CLI packages (src modules + config.py + pyproject.toml).
    agentcore_cli.stage_runtime_code()

    runtime_env = {
        "AWS_REGION": config.AWS_REGION,
        "PROJECT_NAME": config.PROJECT_NAME,
        "RETURNS_KB_ID": config.RETURNS_KB_ID,
        "SHIPPING_KB_ID": config.SHIPPING_KB_ID,
        "WARRANTY_KB_ID": config.WARRANTY_KB_ID,
        "AGENT_LOG_GROUP": config.AGENT_LOG_GROUP,
        "GUARDRAIL_ID": guardrail_id,
        "GUARDRAIL_VERSION": guardrail_version,
    }
    agentcore_cli.configure_runtime(
        env_vars=runtime_env,
        network_mode="PUBLIC",
        protocol="HTTP",
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )
    agentcore_cli.deploy()
    runtime_arn = agentcore_cli.deployed_runtime_arn()

    if not runtime_arn:
        raise RuntimeError("AgentCore CLI deployment did not return a runtime ARN")

    # Wait for the runtime to become READY and return its ARN.
    print(f"  Runtime deployed: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, runtime_arn.split('/')[-1])
    print(' ready.')
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    # TODO: Create AgentCore Memory
    # Use agentcore_control.create_memory() with:
    #   - name (memory_name) and a description
    #   - eventExpiryDuration = 7   (days)
    #   - memoryStrategies = [{'summaryMemoryStrategy': {
    #         'name': 'SessionSummary',
    #         'namespaces': ['/summaries/{actorId}/{sessionId}']}}]
    #   - clientToken (e.g. str(uuid.uuid4())) for idempotency
    # Store the API response in `response`.
    response = agentcore_control.create_memory(
        name=memory_name,
        description="NovaMart customer support session memory",
        eventExpiryDuration=7,
        memoryStrategies=[
            {
                "summaryMemoryStrategy": {
                    "name": "SessionSummary",
                    "namespaces": [
                        "/summaries/{actorId}/{sessionId}"
                    ],
                }
            }
    ],
    clientToken=str(uuid.uuid4()),
)

    if response is None:
        raise NotImplementedError("configure_memory: create_memory() not implemented")

    # Wait until the memory resource is ACTIVE and return its ARN.
    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    if memory['status'] != 'ACTIVE':
        raise TimeoutError(
            f"AgentCore Memory did not become ACTIVE: {memory['status']}"
        )
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    # TODO: Build the logging configuration
    # logging_configuration = {
    #     'cloudWatchConfig': {'logGroupName': config.AGENT_LOG_GROUP,
    #                          'logLevel': 'INFO', 'enabled': True},
    #     'xRayConfig':       {'enabled': True, 'samplingRate': 1.0},
    # }
    # Then apply it:  summary = apply_observability_config(runtime_arn, logging_configuration)
    # Wrap the call in try/except - on success print the CloudWatch log group
    # and the X-Ray sampling rate; on exception print
    #   "[Note] Observability configuration failed: <e>"

    logging_configuration = {
        "cloudWatchConfig": {
            "logGroupName": config.AGENT_LOG_GROUP,
            "logLevel": "INFO",
            "enabled": True,
        },
        "xRayConfig": {
            "enabled": True,
            "samplingRate": 1.0,
        },
    }

    try:
        summary = apply_observability_config(
            runtime_arn,
            logging_configuration,
        )

        print(f"CloudWatch log group: {config.AGENT_LOG_GROUP}")
        print("X-Ray sampling rate: 100%")

    except Exception as e:
        print(f"[Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION = os.environ.get('ORDERS_FUNCTION', '')
_POLICY_FUNCTION = os.environ.get('POLICY_FUNCTION', '')
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', '')


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Optional extension to the in-process @tool functions. Resolve configured
    Lambda functions first; if none exist, skip gateway creation. Otherwise
    create/reuse the gateway and submit its targets. Connecting agents to this
    MCP endpoint requires separate integration; this starter uses in-process tools.

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        A SKIPPED result with a reason, or gateway details and target count.
    """
    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    # Resolve optional Lambda targets before creating any gateway resources.
    available = []
    for target in targets:
        if not target['function']:
            continue
        try:
            available.append((target, _gw_get_function_arn(target['function'])))
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ResourceNotFoundException':
                raise
            print(f"    [Skipped] {target['name']}: Lambda function not found")

    if not available:
        return {'status': 'SKIPPED', 'reason': 'No configured Lambda tool functions are available.'}

    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    print(f"\n  Registering {len(available)} Gateway targets...")
    for target, lambda_arn in available:
        _gw_create_target(agentcore_ctrl, gateway_id, target, lambda_arn)

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url,
            'status': 'TARGETS_SUBMITTED', 'target_count': len(available)}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    # Fail fast if the AgentCore CLI (used by Steps 3 and 5) is missing.
    import agentcore_cli
    print(f"AgentCore CLI: {agentcore_cli.cli_version()} ({agentcore_cli.cli_path()})\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        if gw['status'] == 'SKIPPED':
            print(f"  [Skipped] Gateway: {gw['reason']}")
        else:
            print(f"  Gateway URL : {gw['gateway_url']}")
            print("  Lambda targets submitted; connect an MCP client separately to use them.")
    except Exception as e:
        print(f"  [Note] Optional Gateway deployment failed: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"")
    print(f"  or with the CLI:  agentcore invoke \"What is the return policy for premium customers?\"")
    print(f"  (agentcore status / agentcore logs show the deployed runtime and its logs)\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")
