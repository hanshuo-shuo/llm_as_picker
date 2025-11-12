=== DEBUG Batch 1 INPUT - System ===
You are a top-tier robotics and physics researcher acting as a data evaluation expert. Given multiple 50-step trajectories, select the most valuable ones for training a world model.

=== DEBUG Batch 1 INPUT - User ===
Select exactly 2 trajectories from the list below.
For each selected item, return its episode_id and a one-sentence reason.
Criteria:
- Dynamic Richness: larger state changes and variability imply higher richness.
- Causal Coherence: actions plausibly causing the observed changes.
- Learning Value: usefulness to learn plausible physics and action consequences.

Candidates:
- [1] episode_id=episode_00000
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 5.69944953918457, "avg_reward": 0.11398898810148239, "success_rate": 0.0, "action_magnitude_mean": 2.643186092376709, "action_smoothness_mean": 3.682194709777832, "state_change_magnitude_mean": 74.19131469726562, "height_stats": {"mean": 0.3636663854122162, "std": 0.43511757254600525, "min": 0.08248643577098846, "max": 1.5332199335098267, "start": 1.5332199335098267, "end": 0.09219013154506683}, "com_speed_stats": {"mean": 0.8951735496520996, "std": 1.0845906734466553, "min": 0.0, "max": 4.823164463043213, "start": 0.0, "end": 0.3495577275753021}, "upright_stats": {"mean": 0.03986110910773277, "std": 0.1517849862575531, "min": -0.277953565120697, "max": 0.2496003806591034, "start": 0.17484183609485626, "end": -0.03352850303053856}}
- [2] episode_id=episode_00001
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 5.655882835388184, "avg_reward": 0.11311765760183334, "success_rate": 0.0, "action_magnitude_mean": 2.604212760925293, "action_smoothness_mean": 3.7524006366729736, "state_change_magnitude_mean": 69.91749572753906, "height_stats": {"mean": 0.4184829294681549, "std": 0.41559091210365295, "min": 0.08276347815990448, "max": 1.5194783210754395, "start": 1.5194783210754395, "end": 0.1182015910744667}, "com_speed_stats": {"mean": 0.8276999592781067, "std": 0.9642390608787537, "min": 0.0, "max": 3.8543801307678223, "start": 0.0, "end": 0.2773096561431885}, "upright_stats": {"mean": 0.22874172031879425, "std": 0.2531450688838959, "min": -0.20534054934978485, "max": 0.597553014755249, "start": 0.10251747071743011, "end": 0.085484080016613}}
- [3] episode_id=episode_00002
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 8.530549049377441, "avg_reward": 0.17061097919940948, "success_rate": 0.0, "action_magnitude_mean": 2.592399835586548, "action_smoothness_mean": 3.634506940841675, "state_change_magnitude_mean": 75.22457122802734, "height_stats": {"mean": 0.5270663499832153, "std": 0.4172956645488739, "min": 0.0853816345334053, "max": 1.619579553604126, "start": 1.619579553604126, "end": 0.1002320721745491}, "com_speed_stats": {"mean": 0.8439358472824097, "std": 0.6609410047531128, "min": 0.0, "max": 2.8278253078460693, "start": 0.0, "end": 0.6816325187683105}, "upright_stats": {"mean": 0.3504032492637634, "std": 0.22580420970916748, "min": -0.2695785462856293, "max": 0.7099769115447998, "start": 0.6293657422065735, "end": -0.24702900648117065}}
- [4] episode_id=episode_00003
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 1.7026580572128296, "avg_reward": 0.03405316174030304, "success_rate": 0.0, "action_magnitude_mean": 2.6200826168060303, "action_smoothness_mean": 3.730910062789917, "state_change_magnitude_mean": 77.21348571777344, "height_stats": {"mean": 0.35561394691467285, "std": 0.38535505533218384, "min": 0.0858621746301651, "max": 1.3847472667694092, "start": 1.3847472667694092, "end": 0.2585481107234955}, "com_speed_stats": {"mean": 0.9790966510772705, "std": 1.1686365604400635, "min": 0.0, "max": 4.937502861022949, "start": 0.0, "end": 0.5955928564071655}, "upright_stats": {"mean": 0.002765701152384281, "std": 0.2807071805000305, "min": -0.6115607023239136, "max": 0.4603232443332672, "start": -0.6065936088562012, "end": 0.17239394783973694}}
- [5] episode_id=episode_00004
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 3.9507176876068115, "avg_reward": 0.079014353454113, "success_rate": 0.0, "action_magnitude_mean": 2.629429340362549, "action_smoothness_mean": 3.7112040519714355, "state_change_magnitude_mean": 73.35709381103516, "height_stats": {"mean": 0.33217304944992065, "std": 0.42345407605171204, "min": 0.0861142948269844, "max": 1.515814185142517, "start": 1.515814185142517, "end": 0.08927683532238007}, "com_speed_stats": {"mean": 0.9188806414604187, "std": 1.130277156829834, "min": 0.0, "max": 4.5921759605407715, "start": 0.0, "end": 0.3862375020980835}, "upright_stats": {"mean": -0.20533981919288635, "std": 0.21698182821273804, "min": -0.6668366193771362, "max": 0.11920426785945892, "start": 0.08323246985673904, "end": -0.2943613529205322}}
- [6] episode_id=episode_00005
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 5.345114707946777, "avg_reward": 0.10690229386091232, "success_rate": 0.0, "action_magnitude_mean": 2.6786413192749023, "action_smoothness_mean": 3.6473755836486816, "state_change_magnitude_mean": 76.61161804199219, "height_stats": {"mean": 0.44509029388427734, "std": 0.4125179350376129, "min": 0.10886130481958389, "max": 1.526745319366455, "start": 1.526745319366455, "end": 0.240720734000206}, "com_speed_stats": {"mean": 0.9037848114967346, "std": 1.1015079021453857, "min": 0.0, "max": 4.873201847076416, "start": 0.0, "end": 0.21850724518299103}, "upright_stats": {"mean": 0.20349979400634766, "std": 0.14930459856987, "min": -0.14911068975925446, "max": 0.47916048765182495, "start": 0.14076504111289978, "end": 0.1942683458328247}}
- [7] episode_id=episode_00006
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 1.4502413272857666, "avg_reward": 0.02900482714176178, "success_rate": 0.0, "action_magnitude_mean": 2.612034559249878, "action_smoothness_mean": 3.6205379962921143, "state_change_magnitude_mean": 72.66643524169922, "height_stats": {"mean": 0.306645929813385, "std": 0.38854655623435974, "min": 0.08806867897510529, "max": 1.3984792232513428, "start": 1.3984792232513428, "end": 0.1747463494539261}, "com_speed_stats": {"mean": 1.063761830329895, "std": 1.0786139965057373, "min": 0.0, "max": 4.4340081214904785, "start": 0.0, "end": 0.2409949004650116}, "upright_stats": {"mean": -0.21230363845825195, "std": 0.2961162030696869, "min": -0.853359580039978, "max": 0.21211476624011993, "start": -0.5343201160430908, "end": 0.2111901491880417}}
- [8] episode_id=episode_00007
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 1.7290916442871094, "avg_reward": 0.034581832587718964, "success_rate": 0.0, "action_magnitude_mean": 2.740402936935425, "action_smoothness_mean": 3.9014909267425537, "state_change_magnitude_mean": 77.63751983642578, "height_stats": {"mean": 0.33619046211242676, "std": 0.3971213698387146, "min": 0.08639716356992722, "max": 1.374290108680725, "start": 1.374290108680725, "end": 0.22841544449329376}, "com_speed_stats": {"mean": 0.9430210590362549, "std": 1.0937014818191528, "min": 0.0, "max": 4.437037944793701, "start": 0.0, "end": 0.3901176154613495}, "upright_stats": {"mean": -0.05423169955611229, "std": 0.3118797242641449, "min": -0.661631166934967, "max": 0.41409704089164734, "start": -0.661631166934967, "end": 0.3298223316669464}}
- [9] episode_id=episode_00008
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 3.6364240646362305, "avg_reward": 0.0727284848690033, "success_rate": 0.0, "action_magnitude_mean": 2.6365857124328613, "action_smoothness_mean": 3.770050287246704, "state_change_magnitude_mean": 77.65010833740234, "height_stats": {"mean": 0.3036499321460724, "std": 0.4181993007659912, "min": 0.08142225444316864, "max": 1.4664196968078613, "start": 1.4664196968078613, "end": 0.10722354054450989}, "com_speed_stats": {"mean": 0.8892015814781189, "std": 1.1502468585968018, "min": 0.0, "max": 4.653836727142334, "start": 0.0, "end": 0.3824995756149292}, "upright_stats": {"mean": -0.07178881764411926, "std": 0.10637817531824112, "min": -0.30387023091316223, "max": 0.14465682208538055, "start": -0.17673823237419128, "end": 0.050676483660936356}}
- [10] episode_id=episode_00009
  notes: Significant height drop (possible fall). Low average uprightness.
  metrics: {"length": 50, "total_reward": 11.795119285583496, "avg_reward": 0.23590238392353058, "success_rate": 0.0, "action_magnitude_mean": 2.6726553440093994, "action_smoothness_mean": 3.7934231758117676, "state_change_magnitude_mean": 72.52069854736328, "height_stats": {"mean": 0.7565178871154785, "std": 0.44465479254722595, "min": 0.08298642933368683, "max": 1.6698991060256958, "start": 1.6698991060256958, "end": 0.0912892296910286}, "com_speed_stats": {"mean": 0.7534497976303101, "std": 0.6917417645454407, "min": 0.0, "max": 3.377293348312378, "start": 0.0, "end": 0.3122563064098358}, "upright_stats": {"mean": 0.47539111971855164, "std": 0.430153489112854, "min": -0.40516459941864014, "max": 0.9998332262039185, "start": 0.8942056894302368, "end": -0.08066944032907486}}

Output strictly as JSON:
{
  "selected": [
    {"episode_id": "<id>", "reason": "<one concise sentence>"},
    {"episode_id": "<id>", "reason": "<one concise sentence>"}
  ]
}

The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.

=== DEBUG Batch 1 OUTPUT - Raw ===
Here are the two selected trajectories:

{
  "selected": [
    {"episode_id": "episode_00002", "reason": "High state change magnitude and dynamic richness, indicating a diverse range of physical interactions."},
    {"episode_id": "episode_00009", "reason": "High total reward and success rate, suggesting a well-designed trajectory that explores a wide range of physical phenomena."}
  ]
}

I selected these two trajectories because they exhibit high dynamic richness, which is essential for learning a world model. Episode 00002 has a high state change magnitude, indicating a diverse range of physical interactions, while episode 00009 has a high total reward and success rate, suggesting a well-designed trajectory that explores a wide range of physical phenomena.
