from sft_data import split_pairs, build_example, SEED, N_TEST


def test_split_pairs():
    train, test = split_pairs(SEED, N_TEST)
    assert len(train) == 10000 - N_TEST and len(test) == N_TEST
    assert not set(train) & set(test)  # train 和 test 不能有交集


def test_build_example_loss_positions():
    train, test = split_pairs(SEED, N_TEST)
    for a, b in test:
        x, y = build_example(a, b)
        assert len(x) == len(y)
        n_target = sum(1 for i in y if i != -100)
        assert n_target == len(str(a + b)) + 1
    