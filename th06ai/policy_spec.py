"""Policy compatibility metadata; safe to import from the lightweight GUI."""
POLICY_VERSION = 2
# Per two-frame decision, about 0.15% per second at 30 Hz. A learnable prior,
# not a cap, cooldown, or runtime rule deciding when to bomb.
BOMB_PRIOR = 0.00005
