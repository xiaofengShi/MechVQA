from pathlib import Path

from scripts.upload_latest_ckpts_to_modelscope import (
    _read_access_token_from_upload_script,
    build_upload_plan,
    find_latest_global_step,
    main,
    parse_global_step,
)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parse_global_step_accepts_only_numbered_checkpoint_dirs():
    assert parse_global_step("global_step_327") == 327
    assert parse_global_step("global_step_0009") == 9
    assert parse_global_step("step_10500") == 10500
    assert parse_global_step("global_step_latest") is None
    assert parse_global_step("checkpoint_327") is None


def test_read_access_token_from_upload_script_uses_python_assignments(tmp_path):
    upload_script = tmp_path / "upload_to_modelscope.py"
    upload_script.write_text(
        'ACCESS_TOKEN = "test-token-from-script"\nOTHER = "ignored"\n',
        encoding="utf-8",
    )

    assert _read_access_token_from_upload_script(upload_script) == "test-token-from-script"


def test_read_access_token_from_upload_script_understands_default_constant(tmp_path):
    upload_script = tmp_path / "upload_to_modelscope.py"
    upload_script.write_text(
        "\n".join(
            [
                'DEFAULT_ACCESS_TOKEN = "test-token-from-default"',
                'ACCESS_TOKEN = get_env_value("MODELSCOPE_TOKEN", default=DEFAULT_ACCESS_TOKEN)',
            ]
        ),
        encoding="utf-8",
    )

    assert _read_access_token_from_upload_script(upload_script) == "test-token-from-default"


def test_find_latest_global_step_uses_largest_numeric_suffix(tmp_path):
    experiment = mkdir(tmp_path / "exp_a")
    mkdir(experiment / "global_step_290")
    mkdir(experiment / "global_step_327")
    mkdir(experiment / "global_step_31")
    mkdir(experiment / "step_10500")
    mkdir(experiment / "global_step_latest")

    latest = find_latest_global_step(experiment)

    assert latest == experiment / "step_10500"


def test_build_upload_plan_selects_latest_checkpoint_per_experiment(tmp_path):
    root = mkdir(tmp_path / "JingNeng")
    exp_a = mkdir(root / "jn_qwen3_32b_grpo_n5_0506")
    exp_b = mkdir(root / "jn_qwen3_32b_grpo_n5_stage2_no_replay_p0_p1")
    mkdir(root / "notes_without_checkpoint")
    mkdir(exp_a / "global_step_290")
    mkdir(exp_a / "global_step_327")
    mkdir(exp_b / "global_step_120")
    mkdir(exp_b / "global_step_80")

    plan = build_upload_plan(root, repo_prefix="JingNeng")

    assert [(item.local_path, item.repo_prefix) for item in plan] == [
        (
            exp_a / "global_step_327",
            "JingNeng/jn_qwen3_32b_grpo_n5_0506/global_step_327",
        ),
        (
            exp_b / "global_step_120",
            "JingNeng/jn_qwen3_32b_grpo_n5_stage2_no_replay_p0_p1/global_step_120",
        ),
    ]


def test_build_upload_plan_can_treat_root_as_single_experiment(tmp_path):
    root = mkdir(tmp_path / "jn_qwen3_32b_grpo_n5_0506")
    mkdir(root / "global_step_310")
    mkdir(root / "global_step_327")

    plan = build_upload_plan(root, repo_prefix="JingNeng")

    assert [(item.local_path, item.repo_prefix) for item in plan] == [
        (root / "global_step_327", "JingNeng/jn_qwen3_32b_grpo_n5_0506/global_step_327")
    ]


def test_main_dry_run_prints_plan_without_uploading(tmp_path, capsys):
    root = mkdir(tmp_path / "JingNeng")
    exp = mkdir(root / "jn_qwen3_32b_grpo_n5_0506")
    mkdir(exp / "global_step_327")
    upload_calls = []

    exit_code = main(
        [
            "--root",
            str(root),
            "--repo_id",
            "xiaofengalg/Jn_Heat_CKPT",
            "--repo_prefix",
            "JingNeng",
            "--dry_run",
        ],
        upload_func=lambda **kwargs: upload_calls.append(kwargs),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert upload_calls == []
    assert "DRY RUN" in captured.out
    assert "JingNeng/jn_qwen3_32b_grpo_n5_0506/global_step_327" in captured.out


def test_main_uploads_selected_checkpoints_with_matching_repo_prefixes(tmp_path, monkeypatch):
    root = mkdir(tmp_path / "JingNeng")
    exp_a = mkdir(root / "exp_a")
    exp_b = mkdir(root / "exp_b")
    mkdir(exp_a / "global_step_1")
    mkdir(exp_a / "global_step_3")
    mkdir(exp_b / "global_step_2")
    upload_calls = []
    monkeypatch.setenv("MODELSCOPE_TOKEN", "test-token")

    exit_code = main(
        [
            "--root",
            str(root),
            "--repo_id",
            "xiaofengalg/Jn_Heat_CKPT",
            "--repo_prefix",
            "JingNeng",
            "--repo_type",
            "dataset",
        ],
        upload_func=lambda **kwargs: upload_calls.append(kwargs),
    )

    assert exit_code == 0
    assert upload_calls == [
        {
            "local_path": [str(exp_a / "global_step_3"), str(exp_b / "global_step_2")],
            "repo_id": "xiaofengalg/Jn_Heat_CKPT",
            "token": "test-token",
            "chinese_name": None,
            "visibility": 1,
            "license": "apache-2.0",
            "ignore_patterns": None,
            "include_patterns": None,
            "resume": True,
            "repo_type": "dataset",
            "repo_prefixes": [
                "JingNeng/exp_a/global_step_3",
                "JingNeng/exp_b/global_step_2",
            ],
        }
    ]
