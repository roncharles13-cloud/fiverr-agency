"""LangGraph state machine for the order processing pipeline.

Current shape:

    START
      │
      ▼
    brief_clarification
      │
      ├── halt    → END (operator sends clarification draft)
      └── proceed ▼
                 prompt_engineering
                   │
                   └── route by service_type
                         ├── thumbnail → thumbnail_gen → text_renderer → technical_qc → delivery_packager → END
                         └── other     → END  (placeholder until those agents land)

Class instances are registered via their bound `__call__` method.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agency.agents.background_removal_agent import BackgroundRemovalAgent
from agency.agents.brief_clarification import BriefClarification
from agency.agents.business_design_generator import BusinessDesignGenerator
from agency.agents.delivery_packager import DeliveryPackager
from agency.agents.headshot_generator import HeadshotGenerator
from agency.agents.logo_generator import LogoGenerator
from agency.agents.prompt_engineering import PromptEngineering
from agency.agents.social_graphics_generator import SocialGraphicsGenerator
from agency.agents.technical_qc import TechnicalQC
from agency.agents.text_renderer import TextRenderer
from agency.agents.thumbnail_generator import ThumbnailGenerator
from agency.agents.visual_qc import VisualQC
from agency.state import WorkflowState

NODE_BRIEF_CLARIFICATION = "brief_clarification"
NODE_PROMPT_ENGINEERING = "prompt_engineering"
NODE_THUMBNAIL_GEN = "thumbnail_gen"
NODE_SOCIAL_GRAPHICS_GEN = "social_graphics_gen"
NODE_HEADSHOT_GEN = "headshot_gen"
NODE_BUSINESS_DESIGN_GEN = "business_design"
NODE_LOGO_GEN = "logo_gen"
NODE_BACKGROUND_REMOVAL = "background_removal"
NODE_TEXT_RENDERER = "text_renderer"
NODE_TECHNICAL_QC = "technical_qc"
NODE_VISUAL_QC = "visual_qc"
NODE_DELIVERY_PACKAGER = "delivery_packager"


def build_graph(
    brief_clarification: BriefClarification,
    prompt_engineering: PromptEngineering,
    thumbnail_gen: ThumbnailGenerator,
    social_graphics_gen: SocialGraphicsGenerator,
    headshot_gen: HeadshotGenerator,
    business_design_gen: BusinessDesignGenerator,
    logo_gen: LogoGenerator,
    background_removal: BackgroundRemovalAgent,
    text_renderer: TextRenderer,
    technical_qc: TechnicalQC,
    visual_qc: VisualQC,
    delivery_packager: DeliveryPackager,
) -> CompiledStateGraph:
    """Build and compile the pipeline state machine."""
    graph: StateGraph = StateGraph(WorkflowState)

    graph.add_node(NODE_BRIEF_CLARIFICATION, brief_clarification.__call__)
    graph.add_node(NODE_PROMPT_ENGINEERING, prompt_engineering.__call__)
    graph.add_node(NODE_THUMBNAIL_GEN, thumbnail_gen.__call__)
    graph.add_node(NODE_SOCIAL_GRAPHICS_GEN, social_graphics_gen.__call__)
    graph.add_node(NODE_HEADSHOT_GEN, headshot_gen.__call__)
    graph.add_node(NODE_BUSINESS_DESIGN_GEN, business_design_gen.__call__)
    graph.add_node(NODE_LOGO_GEN, logo_gen.__call__)
    graph.add_node(NODE_BACKGROUND_REMOVAL, background_removal.__call__)
    graph.add_node(NODE_TEXT_RENDERER, text_renderer.__call__)
    graph.add_node(NODE_TECHNICAL_QC, technical_qc.__call__)
    graph.add_node(NODE_VISUAL_QC, visual_qc.__call__)
    graph.add_node(NODE_DELIVERY_PACKAGER, delivery_packager.__call__)

    graph.add_edge(START, NODE_BRIEF_CLARIFICATION)

    # background_removal skips Prompt Engineering — no generation prompt needed.
    graph.add_conditional_edges(
        NODE_BRIEF_CLARIFICATION,
        _route_after_clarification,
        {
            "halt": END,
            "proceed": NODE_PROMPT_ENGINEERING,
            "skip_pe": NODE_BACKGROUND_REMOVAL,
        },
    )

    graph.add_conditional_edges(
        NODE_PROMPT_ENGINEERING,
        _route_to_generation,
        {
            "thumbnail": NODE_THUMBNAIL_GEN,
            "social_graphic": NODE_SOCIAL_GRAPHICS_GEN,
            "headshot": NODE_HEADSHOT_GEN,
            "business_design": NODE_BUSINESS_DESIGN_GEN,
            "logo": NODE_LOGO_GEN,
        },
    )

    # All generators → same editing → QC → packaging tail
    for node in (
        NODE_THUMBNAIL_GEN,
        NODE_SOCIAL_GRAPHICS_GEN,
        NODE_HEADSHOT_GEN,
        NODE_BUSINESS_DESIGN_GEN,
        NODE_LOGO_GEN,
        NODE_BACKGROUND_REMOVAL,
    ):
        graph.add_edge(node, NODE_TEXT_RENDERER)
    graph.add_edge(NODE_TEXT_RENDERER, NODE_TECHNICAL_QC)
    graph.add_edge(NODE_TECHNICAL_QC, NODE_VISUAL_QC)
    graph.add_edge(NODE_VISUAL_QC, NODE_DELIVERY_PACKAGER)
    graph.add_edge(NODE_DELIVERY_PACKAGER, END)

    return graph.compile()


def _route_after_clarification(
    state: WorkflowState,
) -> Literal["halt", "proceed", "skip_pe"]:
    if state.get("clarification_needed"):
        return "halt"
    # Background removal transforms client images — no generation prompt needed.
    if state["service_type"] == "background_removal":
        return "skip_pe"
    return "proceed"


def _route_to_generation(
    state: WorkflowState,
) -> Literal["thumbnail", "social_graphic", "headshot", "logo", "business_design"]:
    # background_removal is routed before reaching this function.
    return state["service_type"]
