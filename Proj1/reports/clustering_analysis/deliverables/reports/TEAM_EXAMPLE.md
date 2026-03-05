# Phase 1: Team Role Embedding Example

## Sample Competitive Team

| Pokemon | Archetype | Type | ATK | DEF | SPD | Role |
|---------|-----------|------|-----|-----|-----|------|
| Koraidon | Generalist | fighting/dragon | 135 | 115 | 135 | Sweeper |
| Miraidon | Speed Sweeper | electric/dragon | 85 | 100 | 135 | Sweeper |
| Groudon | Generalist | ground | 150 | 140 | 90 | Pivot |
| Stakataka | Defensive Wall | rock/steel | 131 | 211 | 13 | Wall |
| Cresselia | Fast Attacker | psychic | 70 | 110 | 85 | Support |
| Gliscor | Generalist | ground/flying | 95 | 125 | 95 | Physical Attacker |


## Team Composition Analysis

**Archetype Diversity**:
- Generalist: 3 Pokemon
- Speed Sweeper: 1 Pokemon
- Defensive Wall: 1 Pokemon
- Fast Attacker: 1 Pokemon


**Strengths**:
- High offensive capability (Sweepers present)
- Good defensive coverage (Wall + Pivot)
- Speed variety for turn order control

**Strategy**: Fast offense with defensive backup; Sweepers + Wall + Pivot balance

## How Team Archetypes Connect to GA Optimization (Phase 2)

In Phase 2, each Pokemon's archetype cluster becomes part of the **fitness function**:

1. **Coverage Check**: GA team must include archetypes that provide stat coverage
2. **Role Synergy**: GA penalizes teams missing defensive walls with fast attackers
3. **Team Balance**: Archetype distribution affects team score (diversity reward)
4. **Weakness Coverage**: Type coverage linked to archetype characteristics

Example: A team with only Speed Sweepers (high speed, low defense) scores poorly on defense,
even if individual Pokemon are strong. GA learns to balance archetypal roles.

---

**Phase 1 Conclusion**: Archetypes established and embedded. Phase 2 uses these cluster
assignments as features to guide GA optimization toward balanced, viable teams.
