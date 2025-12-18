"""
Modern LangGraph Agent for Hierarchical Probability Boosting
Uses LangGraph for state-based agent workflows with langchain 1.0.3
"""

import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from dataclasses import dataclass
import numpy as np
import pandas as pd

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from hierarchical_boosting import (
    HierarchicalConfig, 
    MarkerRelationship, 
    RelationType,
    HierarchicalProbabilityBooster
)


# Define the state for our agent
class AgentState(TypedDict):
    """State definition for the probability boosting agent"""
    messages: Sequence[BaseMessage]
    markers: Optional[List[str]]
    relationships: Optional[List[Dict]]
    config: Optional[Dict]
    dataframe: Optional[Dict]
    results: Optional[Dict]
    current_step: str
    error: Optional[str]


class BiologicalKnowledgeBase:
    """Knowledge base for biological marker relationships"""
    
    KNOWN_RELATIONSHIPS = {
        "T-cell": {
            "primary": ["CD3"],
            "subtypes": ["CD4", "CD8"],
            "exclusive_with": ["CD20", "CD19"]
        },
        "B-cell": {
            "primary": ["CD20", "CD19"],
            "subtypes": ["CD27", "CD38"],
            "exclusive_with": ["CD3"]
        },
        "NK-cell": {
            "primary": ["CD56"],
            "coexpressed": ["CD16"],
            "inhibited_by": ["CD3"]
        },
        "Monocyte": {
            "primary": ["CD14"],
            "coexpressed": ["CD16", "HLA-DR"],
            "subtypes": ["CD16"]
        },
        "Dendritic": {
            "primary": ["CD11c"],
            "coexpressed": ["HLA-DR", "CD123"],
            "exclusive_with": ["CD3", "CD20"]
        }
    }
    
    @classmethod
    def get_marker_relationships(cls, markers: List[str]) -> List[Dict]:
        """Extract known relationships for given markers"""
        relationships = []
        marker_set = set(markers)
        
        for cell_type, info in cls.KNOWN_RELATIONSHIPS.items():
            primary_markers = info.get("primary", [])
            
            for primary in primary_markers:
                if primary in marker_set:
                    # Add subtype relationships
                    subtypes = [s for s in info.get("subtypes", []) if s in marker_set]
                    if subtypes:
                        relationships.append({
                            "parent": primary,
                            "children": subtypes,
                            "type": "subtype",
                            "cell_type": cell_type,
                            "boost_factor": 1.2
                        })
                    
                    # Add exclusivity relationships
                    exclusive = [e for e in info.get("exclusive_with", []) if e in marker_set]
                    if exclusive:
                        relationships.append({
                            "parent": primary,
                            "children": exclusive,
                            "type": "mutually_exclusive",
                            "cell_type": cell_type,
                            "boost_factor": 1.0
                        })
                    
                    # Add coexpression relationships
                    coexpressed = [c for c in info.get("coexpressed", []) if c in marker_set]
                    if coexpressed:
                        relationships.append({
                            "parent": primary,
                            "children": coexpressed,
                            "type": "coexpression",
                            "cell_type": cell_type,
                            "boost_factor": 1.1
                        })
        
        return relationships


# Define tools using the @tool decorator
@tool
def analyze_markers(markers: str) -> str:
    """
    Analyze biological markers and identify their relationships.
    
    Args:
        markers: Comma-separated list of marker names
    
    Returns:
        JSON string with analysis results
    """
    marker_list = [m.strip() for m in markers.split(",")]
    kb = BiologicalKnowledgeBase()
    relationships = kb.get_marker_relationships(marker_list)
    
    # Analyze cell types
    cell_types = list(set([r.get("cell_type", "Unknown") for r in relationships]))
    
    # Generate recommendations
    recommendations = []
    if relationships:
        has_exclusivity = any(r["type"] == "mutually_exclusive" for r in relationships)
        has_subtypes = any(r["type"] == "subtype" for r in relationships)
        
        if has_exclusivity:
            recommendations.append("Use mutual exclusivity to resolve ambiguous classifications")
        if has_subtypes:
            recommendations.append("Leverage subtype markers to boost parent marker confidence")
    else:
        recommendations.append("No known relationships found - consider manual configuration")
    
    return json.dumps({
        "markers": marker_list,
        "relationships": relationships,
        "cell_types": cell_types,
        "recommendations": recommendations
    }, indent=2)


@tool
def design_boosting_config(markers: str, relationships: str) -> str:
    """
    Design a boosting configuration based on markers and relationships.
    
    Args:
        markers: Comma-separated list of marker names
        relationships: JSON string of relationships
    
    Returns:
        JSON string with configuration details
    """
    marker_list = [m.strip() for m in markers.split(",")]
    
    try:
        rel_data = json.loads(relationships) if relationships else []
    except json.JSONDecodeError:
        rel_data = []
    
    # If no relationships provided, auto-detect them
    if not rel_data:
        kb = BiologicalKnowledgeBase()
        rel_data = kb.get_marker_relationships(marker_list)
    
    config = {
        "markers": marker_list,
        "relationships": rel_data,
        "temperature": 0.3,
        "ambiguity_steepness": 5.0,
        "unknown_weight": 0.618,
        "include_unknown": True
    }
    
    return json.dumps({
        "status": "success",
        "config": config,
        "n_relationships": len(rel_data),
        "description": f"Configuration for {len(marker_list)} markers with {len(rel_data)} relationships"
    }, indent=2)


@tool
def execute_boosting(config: str, n_samples: int = 100) -> str:
    """
    Execute probability boosting with given configuration.
    
    Args:
        config: JSON string with configuration
        n_samples: Number of samples to generate for testing
    
    Returns:
        JSON string with execution results
    """
    try:
        config_data = json.loads(config)
        
        # Generate sample data for demonstration
        np.random.seed(42)
        markers = config_data.get("markers", [])
        
        if not markers:
            return json.dumps({
                "status": "error",
                "message": "No markers provided in configuration"
            })
        
        data = {}
        for marker in markers:
            data[f"{marker}_prob"] = np.random.beta(2, 2, n_samples)
        df = pd.DataFrame(data)
        
        # Create configuration
        relationships_data = config_data.get("relationships", [])
        marker_relationships = []
        
        for rel in relationships_data:
            try:
                rel_type = RelationType[rel["type"].upper()]
            except KeyError:
                rel_type = RelationType.MUTUALLY_EXCLUSIVE
                
            marker_relationships.append(
                MarkerRelationship(
                    parent=rel["parent"],
                    children=rel.get("children", []),
                    relation_type=rel_type,
                    boost_factor=rel.get("boost_factor", 1.2),
                    ambiguity_threshold=rel.get("ambiguity_threshold",  0.65)
                )
            )
        
        hier_config = HierarchicalConfig(
            markers=markers,
            relationships=marker_relationships,
            temperature=config_data.get("temperature", 0.3),
            ambiguity_steepness=config_data.get("ambiguity_steepness", 5.0),
            unknown_weight=config_data.get("unknown_weight", 0.618),
            include_unknown=config_data.get("include_unknown", True)
        )
        
        # Execute boosting
        booster = HierarchicalProbabilityBooster(hier_config)
        result_df = booster.process(df)
        
        # Prepare summary
        predictions = result_df["predicted_channel"].value_counts().to_dict()
        
        return json.dumps({
            "status": "success",
            "n_samples": len(result_df),
            "predictions": predictions,
            "mean_confidence": float(result_df["prediction_confidence"].mean()),
            "std_confidence": float(result_df["prediction_confidence"].std()),
            "top_prediction": result_df["predicted_channel"].mode()[0] if len(result_df) > 0 else "unknown"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


class ProbabilityBoostingGraph:
    """
    LangGraph-based agent for probability boosting
    """
    
    def __init__(self, ):
        """Initialize the graph-based agent"""
        
 
        # Create tools list
        self.tools = [analyze_markers, design_boosting_config, execute_boosting]
        self.tool_executor = ToolNode(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Add memory
        self.memory = InMemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("design", self.design_node)
        workflow.add_node("execute", self.execute_node)
        workflow.add_node("summarize", self.summarize_node)
        
        # Set entry point
        workflow.set_entry_point("analyze")
        
        # Add edges
        workflow.add_edge("analyze", "design")
        workflow.add_edge("design", "execute")
        workflow.add_edge("execute", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow
    
    def analyze_node(self, state: AgentState) -> AgentState:
        """Analyze markers and identify relationships"""
        
        markers = state.get("markers", [])
        if not markers:
            # Extract markers from messages if not provided
            last_message = state["messages"][-1] if state["messages"] else None
            if last_message and isinstance(last_message.content, str):
                # Try to extract markers from the message - generic approach
                content = last_message.content
                # Look for comma-separated values or probability columns
                import re
                # Match patterns like "marker1_prob", "marker2_prob" etc.
                prob_pattern = re.findall(r'(\w+)_prob', content)
                if prob_pattern:
                    markers = prob_pattern
                # Or look for comma-separated list
                elif ',' in content:
                    potential_markers = [m.strip() for m in content.split(',')]
                    markers = [m for m in potential_markers if m]
        
        if markers:
            # Use the analyze tool
            result = analyze_markers.invoke({"markers": ",".join(markers)})
            analysis = json.loads(result)
            
            state["relationships"] = analysis.get("relationships", [])
            state["messages"].append(
                AIMessage(content=f"Analyzed {len(markers)} markers. Found {len(state['relationships'])} relationships.")
            )
        else:
            state["messages"].append(
                AIMessage(content="No markers detected. Please provide markers for analysis.")
            )
        
        state["current_step"] = "analyze"
        return state
    
    def design_node(self, state: AgentState) -> AgentState:
        """Design the boosting configuration"""
        
        markers = state.get("markers", [])
        relationships = state.get("relationships", [])
        
        if not markers:
            state["messages"].append(
                AIMessage(content="Cannot design configuration without markers.")
            )
            state["error"] = "No markers provided"
        else:
            # Use the design tool
            result = design_boosting_config.invoke({
                "markers": ",".join(markers),
                "relationships": json.dumps(relationships)
            })
            
            config_result = json.loads(result)
            state["config"] = config_result.get("config", {})
            state["messages"].append(
                AIMessage(content=f"Designed configuration with {config_result.get('n_relationships', 0)} relationships.")
            )
        
        state["current_step"] = "design"
        return state
    
    def execute_node(self, state: AgentState) -> AgentState:
        """Execute the boosting with the configuration"""
        
        config = state.get("config", {})
        
        if config:
            # Use the execute tool
            result = execute_boosting.invoke({
                "config": json.dumps(config),
                "n_samples": 100
            })
            
            execution_result = json.loads(result)
            state["results"] = execution_result
            
            if execution_result.get("status") == "success":
                state["messages"].append(
                    AIMessage(content=f"Boosting completed. Top prediction: {execution_result.get('top_prediction')}. "
                                    f"Mean confidence: {execution_result.get('mean_confidence', 0):.3f}")
                )
            else:
                state["error"] = execution_result.get("message", "Unknown error")
        
        state["current_step"] = "execute"
        return state
    
    def summarize_node(self, state: AgentState) -> AgentState:
        """Summarize the results"""
        
        results = state.get("results", {})
        
        if results and results.get("status") == "success":
            summary = f"""
Probability Boosting Complete!

Results Summary:
- Samples processed: {results.get('n_samples', 0)}
- Mean confidence: {results.get('mean_confidence', 0):.3f}
- Confidence std: {results.get('std_confidence', 0):.3f}

Predictions:
"""
            predictions = results.get('predictions', {})
            for channel, count in predictions.items():
                percentage = (count / results.get('n_samples', 1)) * 100
                summary += f"  - {channel}: {count} ({percentage:.1f}%)\n"
            
            state["messages"].append(AIMessage(content=summary))
        else:
            state["messages"].append(
                AIMessage(content=f"Boosting failed: {state.get('error', 'Unknown error')}")
            )
        
        state["current_step"] = "complete"
        return state
    
    def process(self, 
                markers: List[str],
                df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Process markers and boost probabilities
        
        Args:
            markers: List of marker names
            df: Optional DataFrame with probability columns
        
        Returns:
            Dictionary with results
        """
        
        # Initialize state
        initial_state = {
            "messages": [
                SystemMessage(content="You are a probability boosting agent."),
                HumanMessage(content=f"Boost probabilities for markers: {', '.join(markers)}")
            ],
            "markers": markers,
            "relationships": None,
            "config": None,
            "dataframe": df.to_dict() if df is not None else None,
            "results": None,
            "current_step": "start",
            "error": None
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": "boost-1"}}
        final_state = self.app.invoke(initial_state, config)
        
        return final_state.get("results", {})
    
    def boost_probabilities(self, 
                           df: pd.DataFrame,
                           markers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Main method to boost probabilities
        
        Args:
            df: DataFrame with probability columns
            markers: List of marker names (auto-detected if None)
        
        Returns:
            DataFrame with boosted probabilities
        """
        
        # Auto-detect markers if not provided
        if markers is None:
            prob_columns = [col for col in df.columns if col.endswith('_prob')]
            markers = [col.replace('_prob', '') for col in prob_columns]
        
        if not markers:
            raise ValueError("No markers provided or detected")
        
        # Use the graph to design configuration
        results = self.process(markers, df)
        
        if results.get("status") == "success":
            # Apply the designed configuration to the actual data
            kb = BiologicalKnowledgeBase()
            relationships = kb.get_marker_relationships(markers)
            
            marker_relationships = []
            for rel in relationships:
                try:
                    rel_type = RelationType[rel["type"].upper()]
                except KeyError:
                    rel_type = RelationType.MUTUALLY_EXCLUSIVE
                    
                marker_relationships.append(
                    MarkerRelationship(
                        parent=rel["parent"],
                        children=rel.get("children", []),
                        relation_type=rel_type,
                        boost_factor=rel.get("boost_factor", 1.2),
                        ambiguity_threshold= 0.65
                    )
                )
            
            config = HierarchicalConfig(
                markers=markers,
                relationships=marker_relationships,
                temperature=0.3,
                ambiguity_steepness=5.0,
                unknown_weight=0.618,
                include_unknown=True
            )
            
            booster = HierarchicalProbabilityBooster(config)
            return booster.process(df)
        else:
            raise ValueError(f"Failed to design configuration: {results.get('message', 'Unknown error')}")


def example_usage():
    """Example of using the LangGraph agent"""
    
    print("LangGraph Probability Boosting Agent")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # Create ambiguous cases
    cd3_probs = np.random.beta(2, 2, n_samples)
    cd20_probs = np.random.beta(2, 2, n_samples)
    
    # Make some ambiguous
    ambiguous_mask = np.random.random(n_samples) > 0.7
    cd3_probs[ambiguous_mask] = 0.5 + np.random.normal(0, 0.1, ambiguous_mask.sum())
    cd20_probs[ambiguous_mask] = 0.5 + np.random.normal(0, 0.1, ambiguous_mask.sum())
    
    # Helper markers
    cd4_probs = np.random.beta(1.5, 3, n_samples)
    cd8_probs = np.random.beta(1.5, 3, n_samples)
    
    # Boost helpers when CD3 should win
    cd3_true = cd3_probs > cd20_probs
    cd4_probs[cd3_true & ambiguous_mask] += 0.3
    cd8_probs[cd3_true & ambiguous_mask] += 0.2
    
    df = pd.DataFrame({
        'CD3_prob': np.clip(cd3_probs, 0, 1),
        'CD20_prob': np.clip(cd20_probs, 0, 1),
        'CD4_prob': np.clip(cd4_probs, 0, 1),
        'CD8_prob': np.clip(cd8_probs, 0, 1)
    })
    config = {
    }
    print(f"Created {n_samples} samples with {ambiguous_mask.sum()} ambiguous cases")
    
    # Create agent (API key optional for demo)
    try:
        agent = ProbabilityBoostingGraph()
        print("\nUsing LangGraph agent with Claude...")
        
        # Boost probabilities
        result_df = agent.boost_probabilities(df)
        
        print("\nLangGraph Agent Results:")
        predictions = result_df['predicted_channel'].value_counts()
        for channel, count in predictions.items():
            print(f"  {channel}: {count} ({count/n_samples*100:.1f}%)")
        
        print(f"\nMean confidence: {result_df['prediction_confidence'].mean():.3f}")
        
    except Exception as e:
  
        print(f"Error: {e}")
       
    
    # Check ambiguous cases
    ambiguous_results = result_df[ambiguous_mask]
    print(f"\nAmbiguous cases resolved:")
    for channel, count in ambiguous_results['predicted_channel'].value_counts().items():
        print(f"  {channel}: {count}")
    
    return result_df


if __name__ == "__main__":
    result = example_usage()
    print("\n✅ LangGraph agent working successfully!")
