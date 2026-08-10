from dac_her.ingestion.naming import parse_pdf_name, paper_id_for


def test_main_and_si_variants():
    main = parse_pdf_name("홍기욱_1.pdf")
    assert main is not None
    assert main.role == "main"
    assert main.number == 1

    a = parse_pdf_name("홍기욱_1_SI1.pdf")
    b = parse_pdf_name("홍기욱_1_SI_1.pdf")
    assert a is not None and b is not None
    assert a.si_index == 1 and b.si_index == 1
    assert a.main_filename == "홍기욱_1.pdf"


def test_alias_paper_id():
    assert paper_id_for("홍기욱_12.pdf", "홍기욱", {"홍기욱": "Kiwook"}) == "Kiwook_12"
