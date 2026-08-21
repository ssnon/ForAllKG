import inspect

from domains.dac_her.bridge_validation import bind_bridge_validation, bridge_validation_issues
from domains.dac_her.scientific_signatures import strong_anchor_context_issues


def test_alpha4b2a_generic_validation_accepts_domain_anchor_hook():
    parameter = inspect.signature(bridge_validation_issues).parameters[
        "anchor_context_issues_fn"
    ]
    assert parameter.default is strong_anchor_context_issues


def test_alpha4b2a_bound_validation_callbacks_are_callable():
    def domain_anchor_context_issues(**kwargs):
        del kwargs
        return []

    issues, validate = bind_bridge_validation(domain_anchor_context_issues)
    assert callable(issues)
    assert callable(validate)
