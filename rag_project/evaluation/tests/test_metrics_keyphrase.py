from rag_project.evaluation.metrics import compute_keyphrase_precision


def test_zero_of_four_required_returns_zero():
    answer = "random stuff with no keywords"
    evidence = "doc with keyone keytwo keythree keyfour"
    required = ["keyone", "keytwo", "keythree", "keyfour"]
    assert abs(compute_keyphrase_precision(answer, evidence, required) - 0.0) < 1e-6


def test_two_of_four_answer_both_supported():
    answer = "contains keyone and keytwo in it"
    evidence = "supports keyone keytwo keythree keyfour all present"
    required = ["keyone", "keytwo", "keythree", "keyfour"]
    assert abs(compute_keyphrase_precision(answer, evidence, required) - 0.5) < 1e-6


def test_four_in_answer_three_supported():
    answer = "keyone keytwo keythree keyfour all present here"
    evidence = "contains keyone and keytwo and keythree but NOT_k3yfour_four_written"
    required = ["keyone", "keytwo", "keythree", "keyfour"]
    assert abs(compute_keyphrase_precision(answer, evidence, required) - 0.75) < 1e-6


def test_four_in_answer_all_supported():
    answer = "keyone keytwo keythree keyfour"
    evidence = "keyone keytwo keythree keyfour are all here"
    required = ["keyone", "keytwo", "keythree", "keyfour"]
    assert abs(compute_keyphrase_precision(answer, evidence, required) - 1.0) < 1e-6
