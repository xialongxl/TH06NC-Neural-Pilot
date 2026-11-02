"""Navigation and objective fields, appended without changing old input columns."""
NAV_FIELDS = ('robust_safe','fallback','prefix_risk','action_clearance','action_collision',
              'wall_space','exit_fraction','exit_measured','nominal_collision')
NAV_FEATURES = len(NAV_FIELDS)
DECISION_FIELDS = ('known','fallback','late_control_risk','rescue_opportunity','bomb_allowed',
                   'time_to_contact','observation_age','safe_exit_fraction')
DECISION_FEATURES = len(DECISION_FIELDS)
RESCUE_BOMB_PRIOR = .01
MOBILITY_WEIGHT = .75
EXIT_WEIGHT = .5
EXIT_PROBE_LIMIT = 5
EXIT_TICKS = 6
