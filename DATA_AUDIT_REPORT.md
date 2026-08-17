# Football Player Dataset Audit

## Dataset
- Source file: `players_data-2025_2026.csv`
- Raw shape: 2,839 rows × 102 columns
- Exact duplicate rows: 0
- Duplicate Player–Squad–Competition records: 0

## Position handling
- Multi-position records: 620
- Primary position: first listed position
- Secondary position: second listed position, where available
- Broad groups: Goalkeeper, Defender, Midfielder, Forward

## Playing-time thresholds
- At least 450 minutes: 1,999
- At least 900 minutes: 1,578
- At least 1,350 minutes: 1,192

The baseline model will use 900 minutes because it removes very small samples while retaining enough players for clustering.

## Separate modelling populations
- Eligible outfield players: 1,464
- Eligible goalkeepers: 114

Goalkeepers are separated because their statistics are structurally different from outfield statistics.

## Key transformations
- Trimmed text fields
- Parsed primary and secondary positions
- Added multi-position flag
- Added broad position groups
- Added playing-time eligibility flags
- Calculated selected per-90 rates from raw totals
- Preserved original raw statistics for traceability


## Next Step: Feature Selection and Position-Based Normalisation

The next phase prepares the eligible players for K-Means clustering. The model will not use every available column. It will use only statistics that describe how a player performs on the pitch.

### 1. Excluded from clustering
The following fields will remain in the dataset for identification or display, but they will not be used as K-Means inputs:

- Player name
- Nationality
- Club
- Competition
- Age and year of birth
- Raw position labels
- Matches, starts and minutes
- Eligibility flags
- Market value, reputation or manually assigned ratings

These variables do not directly describe playing style and could bias the clusters.

### 2. Baseline outfield features
The first outfield model will use available rate-based or efficiency statistics covering four performance dimensions:

| Dimension | Candidate features |
|---|---|
| Attacking output | Goals per 90, non-penalty goals per 90, shots per 90, shots on target per 90 |
| Creativity and progression | Assists per 90, crosses per 90, fouls drawn per 90 |
| Defensive activity | Tackles won per 90, interceptions per 90, fouls committed per 90 |
| Discipline and positioning | Yellow cards per 90, red cards per 90, offsides per 90 |

Only features with acceptable completeness, variation and football relevance will be retained.

### 3. Goalkeeper features
Goalkeepers will be modelled separately using available variables such as:

- Goals against per 90
- Save percentage
- Saves per 90
- Clean-sheet percentage
- Penalty saves
- Wins, draws and losses

Goalkeeper rows will never be mixed with outfield-player rows because the underlying performance measures are fundamentally different.

### 4. Position-based standardisation
Selected statistics will be standardised within the relevant broad position group using z-scores:

\[
z = \frac{x - \mu}{\sigma}
\]

where \(x\) is the player's value, \(\mu\) is the mean for the position group and \(\sigma\) is the position-group standard deviation.

This ensures that forwards are compared with forwards, defenders with defenders and midfielders with midfielders before clustering.

### 5. Feature-quality checks
Before K-Means, each candidate variable will be checked for:

- Missing-value percentage
- Near-zero variance
- Extreme skewness
- Impossible values
- Highly correlated duplicates
- Football interpretability

Variables that are mostly missing, constant or redundant will be removed.

### 6. Baseline clustering plan
For each position group, candidate values of \(K\) from 2 to 8 will be tested. The final number of clusters will be selected using:

- Elbow method
- Silhouette score
- Calinski–Harabasz score
- Cluster-size balance
- Stability across random seeds
- Football interpretability of cluster centroids

The final cluster names will be assigned only after examining the statistical profile of each cluster. Examples may include ball-winning midfielder, deep-lying playmaker, direct winger, target forward and defensive full-back.

### 7. Immediate deliverable
The next notebook will produce:

- Final modelling feature list
- Missing-value treatment decisions
- Position-standardised dataset
- K-Means results for candidate values of \(K\)
- Silhouette-score comparison table
- Cluster-centroid profiles
- Initial football interpretation of the clusters



