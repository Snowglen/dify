"""
Unit tests for QuestionClassifierNode._extract_variable_selector_to_variable_mapping method.
"""


from core.workflow.nodes.question_classifier.question_classifier_node import QuestionClassifierNode


class TestExtractVariableSelectorToVariableMapping:
    """Test cases for _extract_variable_selector_to_variable_mapping method."""

    def test_extract_variable_mapping_with_completion_params(self):
        """Test extraction with completion_params containing variable references."""
        node_id = "question_classifier_1"
        node_data = {
            "title": "Test Question Classifier",
            "query_variable_selector": ["start", "query"],
            "instruction": "Classify {{#node1.field#}}",
            "model": {
                "provider": "openai",
                "name": "gpt-3.5-turbo",
                "mode": "completion",
                "completion_params": {
                    "temperature": "{{#node1.temp#}}",
                    "max_tokens": "{{#node1.max#}}",
                },
            },
            "classes": [{"id": "1", "name": "class1"}],
        }

        result = QuestionClassifierNode._extract_variable_selector_to_variable_mapping(
            graph_config={},
            node_id=node_id,
            node_data=node_data,
        )

        assert f"{node_id}.query" in result
        assert f"{node_id}.#node1.field#" in result
        assert f"{node_id}.#node1.temp#" in result
        assert f"{node_id}.#node1.max#" in result
        assert result[f"{node_id}.query"] == ["start", "query"]
        assert result[f"{node_id}.#node1.field#"] == ["node1", "field"]
        assert result[f"{node_id}.#node1.temp#"] == ["node1", "temp"]
        assert result[f"{node_id}.#node1.max#"] == ["node1", "max"]

    def test_extract_variable_mapping_without_completion_params_variables(self):
        """Test extraction without variable references in completion_params."""
        node_id = "question_classifier_1"
        node_data = {
            "title": "Test Question Classifier",
            "query_variable_selector": ["start", "query"],
            "instruction": "Classify {{#node1.field#}}",
            "model": {
                "provider": "openai",
                "name": "gpt-3.5-turbo",
                "mode": "completion",
                "completion_params": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
            },
            "classes": [{"id": "1", "name": "class1"}],
        }

        result = QuestionClassifierNode._extract_variable_selector_to_variable_mapping(
            graph_config={},
            node_id=node_id,
            node_data=node_data,
        )

        assert f"{node_id}.query" in result
        assert f"{node_id}.#node1.field#" in result
        assert f"{node_id}.#node1.temp#" not in result
        assert f"{node_id}.#node1.max#" not in result

    def test_extract_variable_mapping_with_empty_completion_params(self):
        """Test extraction with empty completion_params."""
        node_id = "question_classifier_1"
        node_data = {
            "title": "Test Question Classifier",
            "query_variable_selector": ["start", "query"],
            "instruction": "Classify the question",
            "model": {
                "provider": "openai",
                "name": "gpt-3.5-turbo",
                "mode": "completion",
                "completion_params": {},
            },
            "classes": [{"id": "1", "name": "class1"}],
        }

        result = QuestionClassifierNode._extract_variable_selector_to_variable_mapping(
            graph_config={},
            node_id=node_id,
            node_data=node_data,
        )

        assert f"{node_id}.query" in result
        assert len(result) == 1
