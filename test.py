import pandas as pd

# Load your DataFrame
historic_scores_df = pd.read_csv('historic_scores.csv')

# Ensure 'puzzle_number' is of type int
historic_scores_df['puzzle_number'] = historic_scores_df['puzzle_number'].astype(int)

# Manually check for a duplicate
user = 'U061F5HE13J'
puzzle_number = 123

is_duplicate = (
    (historic_scores_df['user'] == user) &
    (historic_scores_df['puzzle_number'] == puzzle_number)
).any()

print(f"Is duplicate: {is_duplicate}")

if is_duplicate:
    print("Actual entries:")
    print(historic_scores_df[(historic_scores_df['user'] == user) &
                             (historic_scores_df['puzzle_number'] == puzzle_number)])
else:
    print("No duplicate entries found.")