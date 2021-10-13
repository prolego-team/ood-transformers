import nlp_datasets

def test_reuters_dataset_dictionaries():
    """

    """
    reuters_data = nlp_datasets.reuters_dataset_dictionaries()
    assert type(reuters_data) == list
    assert len(reuters_data) == 10788
    assert type(reuters_data[0]) == dict
    for key in ["guid", "text", "labels", "is_train"]:
        assert key in reuters_data[0].keys()
