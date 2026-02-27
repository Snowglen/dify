"""
Unit tests for AgentNode._extract_variable_selector_to_variable_mapping method.
"""

from core.workflow.nodes.agent.agent_node import AgentNode


class TestExtractVariableSelectorToVariableMapping:
    """Test cases for _extract_variable_selector_to_variable_mapping method."""

    def test_extract_variable_mapping_with_completion_params_in_model_selector(self):
        """Test extraction with completion_params in model selector containing variable references."""
        node_id = "agent_node_1"
        node_data = {
            "title": "Test Agent",
            "agent_strategy_provider_name": "test_provider",
            "agent_strategy_name": "test_strategy",
            "agent_strategy_label": "test_label",
            "agent_parameters": {
                "model": {
                    "type": "mixed",
                    "value": {
                        "provider": "openai",
                        "name": "gpt-3.5-turbo",
                        "completion_params": {
                            "temperature": "{{#node1.temp#}}",
                            "max_tokens": "{{#node1.max#}}",
                        },
                    },
                },
            },
            "output_schema": {},
        }

        result = AgentNode._extract_variable_selector_to_variable_mapping(
            graph_config={},
            node_id=node_id,
            node_data=node_data,
        )

        assert f"{node_id}.#node1.temp#" in result
        assert f"{node_id}.#node1.max#" in result
        assert result[f"{node_id}.#node1.temp#"] == ["node1", "temp"]
        assert result[f"{node_id}.#node1.max#"] == ["node1", "max"]

    def test_extract_variable_mapping_with_string_template(self):
        """Test extraction with string template in agent_parameters."""
        node_id = "agent_node_1"
        node_data = {
            "title": "Test Agent",
            "agent_strategy_provider_name": "test_provider",
            "agent_strategy_name": "test_strategy",
            "agent_strategy_label": "test_label",
            "agent_parameters": {
                "prompt": {
                    "type": "mixed",
                    "value": "Hello {{#node1.name#}}",
                },
            },
            "output_schema": {},
        }

        result = AgentNode._extract_variable_selector_to_variable_mapping(
            graph_config={},
            node_id=node_id,
            node_data=node_data,
        )

        assert f"{node_id}.#node1.name#" in result
        assert result[f"{node_id}.#node1.name#"] == ["node1", "name"]

    def test_extract_variable_mapping_with_variable_type_parameter(self):
        """Test extraction with variable type parameter."""
        node_id = "agent_node_1"
        node_data = {
            "title": "Test Agent",
            "agent_strategy_provider_name": "test_provider",
            "agent_strategy_name": "test_strategy",
            "agent_strategy_label": "test_label",
            "agent_parameters": {
                "query": {
                    "type": "variable",
                    "value": ["start", "query"],
                },
            },
            "output_schema": {},
        }

        result = AgentNode._extract_variable_selector_to_variable_mapping(
            graph_config={},
            node_id=node_id,
            node_data=node_data,
        )

        assert f"{node_id}.query" in result
        assert result[f"{node_id}.query"] == ["start", "query"]

    def test_extract_variable_mapping_with_mixed_parameters(self):
        """Test extraction with mixed parameter types."""
        node_id = "agent_node_1"
        node_data = {
            "title": "Test Agent",
            "agent_strategy_provider_name": "test_provider",
            "agent_strategy_name": "test_strategy",
            "agent_strategy_label": "test_label",
            "agent_parameters": {
                "query": {
                    "type": "variable",
                    "value": ["start", "query"],
                },
                "prompt": {
                    "type": "mixed",
                    "value": "Hello {{#node1.name#}}",
                },
                "model": {
                    "type": "mixed",
                    "value": {
                        "provider": "openai",
                        "name": "gpt-3.5-turbo",
                        "completion_params": {
                            "temperature": "{{#node1.temp#}}",
                        },
                    },
                },
            },
            "output_schema": {},
        }

        result = AgentNode._extract_variable_selector_to_variable_mapping(
            graph_config={},
            node_id=node_id,
            node_data=node_data,
        )

        assert f"{node_id}.query" in result
        assert f"{node_id}.#node1.name#" in result
        assert f"{node_id}.#node1.temp#" in result
        assert result[f"{node_id}.query"] == ["start", "query"]
        assert result[f"{node_id}.#node1.name#"] == ["node1", "name"]
        assert result[f"{node_id}.#node1.temp#"] == ["node1", "temp"]

    def test_extract_variable_mapping_with_empty_agent_parameters(self):
        """Test extraction with empty agent_parameters."""
        node_id = "agent_node_1"
        node_data = {
            "title": "Test Agent",
            "agent_strategy_provider_name": "test_provider",
            "agent_strategy_name": "test_strategy",
            "agent_strategy_label": "test_label",
            "agent_parameters": {},
            "output_schema": {},
        }

        result = AgentNode._extract_variable_selector_to_variable_mapping(
            graph_config={},
            node_id=node_id,
            node_data=node_data,
        )

        assert result == {}
