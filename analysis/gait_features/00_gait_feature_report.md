# 多步态特征体检报告

本报告由 APEX imitation motion CSV 自动生成，用于在正式训练前判断不同 gait family 的运动特征差异。

## 数据集概况

- 分析 motion 数量：46
- gait family 计数：{'walk': 17, 'trot': 9, 'jump': 4, 'run': 3, 'lateral': 3, 'turn': 3, 'canter': 2, 'pace': 2, 'crawl': 2, 'unknown': 1}

## 第一版多步态建模建议

- 先不要引入情绪标签，优先把 walk / trot / pace / canter / bound 等 gait 的周期结构做对。
- CPG 负责提供相位、频率、基础关节轨迹；RL 后续只学习 residual 或少量 CPG 参数修正。
- `gait_family` 来自 motion 文件名关键词，只用于整理数据，不代表已完成接触序列识别。

## gait family 均值

| gait_family   |   motions |   speed_abs_mean |   step_frequency_hz |   foot_clearance_mean |   body_bounce_std |
|:--------------|----------:|-----------------:|--------------------:|----------------------:|------------------:|
| canter        |         2 |           1.4491 |              2.0285 |                0.1734 |            0.0538 |
| crawl         |         2 |           0.0000 |              2.3125 |                0.1206 |            0.0550 |
| jump          |         4 |           0.5988 |              2.3689 |                0.1597 |            0.0490 |
| lateral       |         3 |           0.5680 |              1.4028 |                0.0897 |            0.0135 |
| pace          |         2 |           0.9164 |              2.3207 |                0.1106 |            0.0308 |
| run           |         3 |           1.2167 |              1.4174 |                0.1711 |            0.0578 |
| trot          |         9 |           0.7322 |              1.5060 |                0.0667 |            0.0266 |
| turn          |         3 |           0.2783 |              1.2861 |                0.0923 |            0.0324 |
| unknown       |         1 |           0.0000 |              2.5000 |                0.1414 |            0.0520 |
| walk          |        17 |           0.5591 |              1.4338 |                0.0830 |            0.0277 |

## 全量 motion 指标

| motion                            | source_dir   | gait_family   |   speed_abs_mean |   step_frequency_hz |   foot_clearance_mean |   body_height_mean |   body_bounce_std |   contact_duty_factor_mean |   foot_slip_speed_mean |   joint_limit_violation_count |
|:----------------------------------|:-------------|:--------------|-----------------:|--------------------:|----------------------:|-------------------:|------------------:|---------------------------:|-----------------------:|------------------------------:|
| go2_retarget_canter_2ms           | animal_mocap | canter        |           2.0237 |              2.9471 |                0.1629 |             0.2513 |            0.0192 |                     0.3007 |               nan      |                             0 |
| ai4_dog_canter                    | kine2go      | canter        |           0.8746 |              1.1099 |                0.1839 |             0.2136 |            0.0884 |                     0.3002 |                 2.1397 |                             0 |
| solo8_crawl_slow                  | kine2go      | crawl         |           0.0000 |              1.7500 |                0.1378 |             0.1544 |            0.0691 |                     0.3000 |                 0.2836 |                             0 |
| solo8_crawl_fast                  | kine2go      | crawl         |           0.0000 |              2.8750 |                0.1034 |             0.2152 |            0.0409 |                     0.3000 |                 0.9870 |                             0 |
| go2_retarget_jump                 | animal_mocap | jump          |           1.5122 |              2.3227 |                0.2812 |             0.2877 |            0.0345 |                     0.3007 |               nan      |                             0 |
| ai4_dog_synth_half_flip_jump      | kine2go      | jump          |           0.8830 |              1.6531 |                0.0526 |             0.3240 |            0.0039 |                     0.3001 |                 0.2941 |                             0 |
| solo8_jump_forward_b              | kine2go      | jump          |           0.0000 |              2.5000 |                0.1535 |             0.2449 |            0.0823 |                     0.3000 |                 2.1122 |                             0 |
| solo8_jump_forward_a              | kine2go      | jump          |           0.0000 |              3.0000 |                0.1515 |             0.2690 |            0.0752 |                     0.3000 |                 2.5197 |                             0 |
| ai4_dog_synth_wide_strafe         | kine2go      | lateral       |           0.9709 |              1.6812 |                0.0610 |             0.3186 |            0.0056 |                     0.2998 |                 0.3474 |                             0 |
| ai4_dog_synth_tight_strafe        | kine2go      | lateral       |           0.5526 |              1.4418 |                0.0531 |             0.3213 |            0.0047 |                     0.2999 |                 0.2616 |                             0 |
| sidesteps_go1_STMR_resampled_50Hz | animal_mocap | lateral       |           0.1805 |              1.0855 |                0.1550 |             0.3245 |            0.0301 |                     0.3000 |               nan      |                             0 |
| go2_retarget_pace                 | animal_mocap | pace          |           1.0331 |              3.0220 |                0.0419 |             0.3145 |            0.0060 |                     0.3007 |               nan      |                             0 |
| ai4_dog_pace                      | kine2go      | pace          |           0.7997 |              1.6194 |                0.1792 |             0.3038 |            0.0555 |                     0.2996 |                 0.2388 |                             0 |
| ai4_dog_run_02                    | kine2go      | run           |           2.4389 |              2.1450 |                0.1623 |             0.3349 |            0.0159 |                     0.3018 |                 0.8627 |                             0 |
| ai4_dog_run_00                    | kine2go      | run           |           0.8746 |              1.1099 |                0.1839 |             0.2136 |            0.0884 |                     0.3002 |                 2.1397 |                             0 |
| ai4_dog_run_04                    | kine2go      | run           |           0.3367 |              0.9972 |                0.1670 |             0.1887 |            0.0691 |                     0.3001 |                 1.0721 |                             0 |
| go2_retarget_trot                 | animal_mocap | trot          |           1.4772 |              2.4226 |                0.1763 |             0.2872 |            0.0103 |                     0.3007 |               nan      |                             0 |
| ai4_dog_trot_00                   | kine2go      | trot          |           0.8416 |              1.5591 |                0.0772 |             0.3221 |            0.0098 |                     0.2998 |                 0.0771 |                             0 |
| vhdc_horse1_s1_trot_02            | kine2go      | trot          |           0.6599 |              1.7000 |                0.0297 |             0.3353 |            0.0332 |                     0.3000 |                 0.1586 |                             0 |
| vhdc_horse1_s1_trot_01            | kine2go      | trot          |           0.6597 |              1.6000 |                0.0300 |             0.3436 |            0.0350 |                     0.3000 |                 0.1569 |                             0 |
| vhdc_horse1_s2_trot_03            | kine2go      | trot          |           0.6568 |              1.1750 |                0.0297 |             0.3120 |            0.0174 |                     0.3000 |                 0.0695 |                             0 |
| vhdc_horse1_s1_trot_03            | kine2go      | trot          |           0.6547 |              1.8250 |                0.0286 |             0.3355 |            0.0314 |                     0.3000 |                 0.1619 |                             0 |
| vhdc_horse1_s2_trot_01            | kine2go      | trot          |           0.6524 |              1.1500 |                0.0305 |             0.3109 |            0.0157 |                     0.3000 |                 0.0696 |                             0 |
| vhdc_horse1_s2_trot_02            | kine2go      | trot          |           0.6505 |              1.1250 |                0.0310 |             0.3131 |            0.0176 |                     0.3000 |                 0.0634 |                             0 |
| ai4_dog_trot_01                   | kine2go      | trot          |           0.3367 |              0.9972 |                0.1670 |             0.1887 |            0.0691 |                     0.3001 |                 1.0721 |                             0 |
| ai4_dog_left_turn                 | kine2go      | turn          |           0.3976 |              1.2386 |                0.0754 |             0.3207 |            0.0192 |                     0.3000 |                 0.1060 |                             0 |
| ai4_dog_right_turn                | kine2go      | turn          |           0.3976 |              1.2386 |                0.0754 |             0.3207 |            0.0192 |                     0.3000 |                 0.1060 |                             0 |
| hopturn_go2_STMR_resampled_50Hz   | animal_mocap | turn          |           0.0398 |              1.3812 |                0.1260 |             0.2930 |            0.0590 |                     0.3039 |               nan      |                             0 |
| solo8_scoot_forward               | kine2go      | unknown       |           0.0000 |              2.5000 |                0.1414 |             0.2529 |            0.0520 |                     0.3000 |                 1.1977 |                             0 |
| ai4_dog_walk_04                   | kine2go      | walk          |           0.9434 |              0.9573 |                0.1763 |             0.2528 |            0.0834 |                     0.2998 |                 0.6621 |                             0 |
| ai4_dog_synth_square_walk         | kine2go      | walk          |           0.9274 |              1.7204 |                0.0555 |             0.3235 |            0.0051 |                     0.3000 |                 0.3103 |                             0 |
| ai4_dog_synth_circle_walk         | kine2go      | walk          |           0.9154 |              1.6345 |                0.0637 |             0.3190 |            0.0059 |                     0.3001 |                 0.3088 |                             0 |
| ai4_dog_synth_ellipse_walk        | kine2go      | walk          |           0.8830 |              1.6531 |                0.0526 |             0.3240 |            0.0039 |                     0.3001 |                 0.2941 |                             0 |
| ai4_dog_walk_03                   | kine2go      | walk          |           0.8416 |              1.5591 |                0.0772 |             0.3221 |            0.0098 |                     0.2998 |                 0.0771 |                             0 |
| ai4_dog_walk_00                   | kine2go      | walk          |           0.7997 |              1.6194 |                0.1792 |             0.3038 |            0.0555 |                     0.2996 |                 0.2388 |                             0 |
| ai4_dog_walk_02                   | kine2go      | walk          |           0.5766 |              1.2681 |                0.1714 |             0.2642 |            0.0772 |                     0.3004 |                 0.2762 |                             0 |
| ai4_dog_walk_01                   | kine2go      | walk          |           0.5722 |              1.5660 |                0.1796 |             0.2808 |            0.0766 |                     0.2998 |                 0.2198 |                             0 |
| ai4_dog_synth_eight_walk          | kine2go      | walk          |           0.5523 |              1.5561 |                0.0541 |             0.3219 |            0.0041 |                     0.3000 |                 0.3123 |                             0 |
| vhdc_horse1_s2_walk_01            | kine2go      | walk          |           0.3945 |              0.9625 |                0.0237 |             0.2927 |            0.0066 |                     0.3000 |                 0.0204 |                             0 |
| vhdc_horse1_s2_walk_02            | kine2go      | walk          |           0.3875 |              1.0125 |                0.0231 |             0.2933 |            0.0062 |                     0.3000 |                 0.0201 |                             0 |
| vhdc_horse1_s2_walk_03            | kine2go      | walk          |           0.3818 |              0.9750 |                0.0236 |             0.2925 |            0.0076 |                     0.3000 |                 0.0256 |                             0 |
| vhdc_horse1_s1_walk_01            | kine2go      | walk          |           0.3499 |              1.6000 |                0.0203 |             0.2935 |            0.0088 |                     0.3000 |                 0.0519 |                             0 |
| vhdc_horse1_s1_walk_03            | kine2go      | walk          |           0.3468 |              1.5250 |                0.0212 |             0.2950 |            0.0097 |                     0.3000 |                 0.0476 |                             0 |
| vhdc_horse1_s1_walk_02            | kine2go      | walk          |           0.3437 |              1.5250 |                0.0202 |             0.2909 |            0.0086 |                     0.3000 |                 0.0461 |                             0 |
| ai4_dog_walk_06                   | kine2go      | walk          |           0.2893 |              0.9907 |                0.2157 |             0.1539 |            0.0846 |                     0.3001 |                 0.9327 |                             0 |
| solo8_walk                        | kine2go      | walk          |           0.0000 |              2.2500 |                0.0531 |             0.3140 |            0.0164 |                     0.3000 |                 0.1811 |                             0 |

## 注意事项

- contact pattern 目前由足端高度低位区间近似得到，因为 CSV 没有显式触地标签。
- foot slip 只在存在 world foot position 列时计算；缺失时记为 `nan`。
- joint limit violation 目前使用保守的 CSV 空间宽限位，只作为第一轮数据健康检查。
- 后续做真实 Go2 安全边界时，应切换到 URDF 精确关节限位。
