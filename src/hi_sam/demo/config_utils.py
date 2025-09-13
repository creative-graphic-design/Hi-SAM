def get_detection_params(dataset: str, zero_shot: bool = False):
    """Get detection parameters based on dataset and mode."""
    if dataset == "totaltext":
        if zero_shot:
            return {
                "fg_points_num": 50,
                "score_thresh": 0.3,
                "unclip_ratio": 1.5
            }
        else:
            return {
                "fg_points_num": 500,
                "score_thresh": 0.95
            }
    elif dataset == "ctw1500":
        if zero_shot:
            return {
                "fg_points_num": 100,
                "score_thresh": 0.6
            }
        else:
            return {
                "fg_points_num": 300,
                "score_thresh": 0.7
            }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")