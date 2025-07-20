import pandas as pd
def predictions_to_competition_df(
    predictions: pd.DataFrame, test_date: str = "2025-04-28", 
    date_before_test: str = "2025-04-25",
    test_data_for_autogluon: pd.DataFrame = None,
    model_mode: str = "price_change"
) -> pd.DataFrame:
    prediction_preds = predictions.reset_index()
    only_28 = prediction_preds[prediction_preds["timestamp"] == test_date][
        ["item_id", "mean"]
    ]
    if model_mode == "price_change":
        sorted_df = only_28.sort_values(by="mean", ascending=False)
    elif model_mode == "-price_change":
        sorted_df = only_28.sort_values(by="mean", ascending=True)
    else:
        # prediction_preds
        val_true = test_data_for_autogluon.reset_index()
        only_25 = val_true[val_true["timestamp"] == date_before_test][["item_id", "收盘"]]
        # only_25

        combine25and28 = pd.merge(only_25, only_28, on="item_id")
        combine25and28["涨幅"] = (
            (combine25and28["mean"] - combine25and28["收盘"])
            / combine25and28["收盘"]
            * 100
        )
        # combine25and28

        # --- 排序并提取前10和后10的item_id ---
        # 按涨幅排序（降序）
        sorted_df = combine25and28.sort_values(by="涨幅", ascending=False)
    # %%
    # 提取涨幅最大的前10个item_id
    top_10_item_ids = sorted_df["item_id"].head(10).astype(int).tolist()
    # 提取涨幅最小的后10个item_id
    bottom_10_item_ids = sorted_df["item_id"].tail(10).astype(int).tolist()
    # --- 创建新的DataFrame ---
    result_df = pd.DataFrame(
        {"涨幅最大股票代码": top_10_item_ids, "涨幅最小股票代码": bottom_10_item_ids}
    )
    return result_df