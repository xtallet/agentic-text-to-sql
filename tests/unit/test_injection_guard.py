from app.domain.prompts.injection_guard import wrap_untrusted


def test_wraps_content_in_a_tag_pair():
    result = wrap_untrusted("no such table: Users")

    assert "no such table: Users" in result
    assert "<untrusted_data_" in result
    assert "</untrusted_data_" in result


def test_includes_ignore_instructions_directive():
    result = wrap_untrusted("some content")

    assert "ignore" in result.lower()
    assert "untrusted data" in result.lower()


def test_tag_is_randomized_between_calls():
    first = wrap_untrusted("same content")
    second = wrap_untrusted("same content")

    assert first != second


def test_custom_label_is_used_in_tag():
    result = wrap_untrusted("some content", label="db_error")

    assert "<db_error_" in result
