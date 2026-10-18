
"""
LearnIQ: Synthetic Student Performance Dataset Generator

Generates:
1. student_performance_messy.csv
2. student_performance_clean_reference.csv
3. topic_attempts_messy.csv

All records are synthetic and do not represent real students.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
N_STUDENTS = 5500
TOPICS_PER_STUDENT = 8

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw"
)

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clip(values, lower, upper):
    """Keep numeric values within a realistic range."""
    return np.clip(values, lower, upper)


def add_missing_values(df, columns, rate=0.025):
    """Introduce missing values in selected columns."""
    df = df.copy()

    for column in columns:
        mask = rng.random(len(df)) < rate
        df.loc[mask, column] = np.nan

    return df


def mess_up_categories(series, probability=0.12):
    """Introduce realistic capitalization and whitespace issues."""
    result = series.astype(object).copy()

    variations = {
        "Good": ["good", " GOOD", "Good ", "GOOD"],
        "Average": ["average", " Average ", "AVERAGE"],
        "Poor": ["poor", " POOR", "Poor "],
        "High": ["high", " HIGH", "High "],
        "Medium": ["medium", " Medium ", "MEDIUM"],
        "Low": ["low", " LOW", "Low "],
        "Visual": ["visual", " VISUAL", "Visual "],
        "Reading": ["reading", " READING", "Reading "],
        "Practice": ["practice", " PRACTICE", "Practice "],
        "Auditory": ["auditory", " AUDITORY", "Auditory "],
        "Yes": ["yes", " YES", "Yes "],
        "No": ["no", " NO", "No "],
    }

    for index in result.index:
        value = result.at[index]

        if (
            pd.notna(value)
            and value in variations
            and rng.random() < probability
        ):
            result.at[index] = rng.choice(variations[value])

    return result


# ============================================================
# STUDENT-LEVEL DATA GENERATION
# ============================================================

def generate_students(n=N_STUDENTS):
    # Latent learning profiles make the data less random.
    profiles = rng.choice(
        ["consistent", "average", "struggling"],
        size=n,
        p=[0.35, 0.45, 0.20],
    )

    profile_effect = np.select(
        [
            profiles == "consistent",
            profiles == "average",
            profiles == "struggling",
        ],
        [12.0, 0.0, -13.0],
    )

    # Synthetic student identifiers.
    student_ids = [
        f"STU{i:05d}" for i in range(1, n + 1)
    ]

    age = rng.integers(17, 25, size=n)

    grade_level = rng.choice(
        ["Year 1", "Year 2", "Year 3", "Year 4"],
        size=n,
        p=[0.30, 0.28, 0.24, 0.18],
    )

    # Academic history and learning behavior.
    previous_marks = clip(
        rng.normal(62 + profile_effect * 0.8, 14, n),
        12,
        98,
    )

    attendance = clip(
        rng.normal(
            79 + profile_effect * 0.65,
            12,
            n,
        ),
        30,
        100,
    )

    study_hours = clip(
        rng.gamma(shape=2.3, scale=2.0, size=n)
        + np.where(profiles == "consistent", 2.0, 0.0)
        - np.where(profiles == "struggling", 0.6, 0.0),
        0,
        35,
    )

    study_days = clip(
        np.rint(
            study_hours / 5
            + rng.normal(1.0, 1.0, n)
        ),
        0,
        7,
    ).astype(int)

    sleep_hours = clip(
        rng.normal(7.0, 1.15, n),
        3.0,
        11.0,
    )

    study_frequency = np.select(
        [
            study_days >= 5,
            study_days >= 3,
        ],
        ["High", "Medium"],
        default="Low",
    )

    study_consistency = clip(
        rng.beta(5, 2, n) * 100
        + np.where(profiles == "consistent", 5, 0)
        - np.where(profiles == "struggling", 10, 0),
        0,
        100,
    )

    # Assessment scores share a common academic signal,
    # with independent noise for each assessment.
    academic_signal = (
        0.38 * previous_marks
        + 0.22 * attendance
        + 1.15 * study_hours
        + 0.12 * study_consistency
        + profile_effect
    )

    quiz_average = clip(
        academic_signal * 0.62
        + rng.normal(17, 10, n),
        0,
        100,
    )

    assignment_average = clip(
        academic_signal * 0.57
        + rng.normal(22, 9, n),
        0,
        100,
    )

    missed_assignments = clip(
        np.rint(
            rng.poisson(
                np.maximum(0.25, 2.5 - study_days * 0.28),
                size=n,
            )
        ),
        0,
        10,
    ).astype(int)

    average_attempts = clip(
        rng.poisson(
            np.maximum(0.6, 2.4 - quiz_average / 75),
            size=n,
        ) + 1,
        1,
        8,
    ).astype(int)

    learning_style = rng.choice(
        ["Visual", "Reading", "Practice", "Auditory"],
        size=n,
        p=[0.30, 0.20, 0.35, 0.15],
    )

    internet_access = rng.choice(
        ["Yes", "No"],
        size=n,
        p=[0.91, 0.09],
    )

    # Final marks are a synthetic outcome influenced by
    # earlier performance and learning behavior.
    raw_final = (
        0.30 * previous_marks
        + 0.24 * quiz_average
        + 0.20 * assignment_average
        + 0.13 * attendance
        + 0.75 * study_hours
        + 0.08 * study_consistency
        - 1.4 * missed_assignments
        + rng.normal(0, 7.5, n)
    )

    final_marks = clip(raw_final, 0, 100)
    final_marks = np.round(final_marks, 1)

    passed = (final_marks >= 40).astype(int)

    performance_category = np.select(
        [
            final_marks >= 75,
            final_marks >= 60,
            final_marks >= 40,
        ],
        ["Excellent", "Good", "Average"],
        default="Poor",
    )

    # Risk labels are based on final outcomes for this
    # synthetic dataset. Exclude outcome-derived columns
    # when training a model to predict risk.
    risk_score = (
        0.40 * (100 - attendance)
        + 0.35 * (100 - quiz_average)
        + 0.25 * (100 - previous_marks)
    )

    risk_level = np.select(
        [
            (final_marks < 40) | (risk_score >= 60),
            (final_marks < 55) | (risk_score >= 45),
        ],
        ["High", "Medium"],
        default="Low",
    )

    df = pd.DataFrame({
        "student_id": student_ids,
        "age": age,
        "grade_level": grade_level,
        "previous_marks": np.round(previous_marks, 1),
        "attendance_pct": np.round(attendance, 1),
        "study_hours_per_week": np.round(study_hours, 1),
        "study_days_per_week": study_days,
        "study_consistency_pct": np.round(study_consistency, 1),
        "sleep_hours_per_night": np.round(sleep_hours, 1),
        "quiz_average": np.round(quiz_average, 1),
        "assignment_average": np.round(assignment_average, 1),
        "missed_assignments": missed_assignments,
        "average_attempts": average_attempts,
        "study_frequency": study_frequency,
        "learning_style": learning_style,
        "internet_access": internet_access,
        "final_marks": final_marks,
        "passed": passed,
        "performance_category": performance_category,
        "risk_level": risk_level,
    })

    return df


# ============================================================
# TOPIC-LEVEL DATA GENERATION
# ============================================================

def generate_topic_attempts(students):
    topics = [
        ("Python Basics", 1),
        ("Arrays", 2),
        ("Linked Lists", 3),
        ("Stacks and Queues", 3),
        ("Trees", 4),
        ("Graphs", 5),
        ("Sorting Algorithms", 3),
        ("Searching Algorithms", 2),
    ]

    records = []

    for student in students.itertuples(index=False):
        student_ability = (
            0.30 * student.previous_marks
            + 0.25 * student.quiz_average
            + 0.20 * student.assignment_average
            + 0.25 * student.attendance_pct
        )

        for topic, difficulty in topics:
            # Harder topics tend to have lower scores.
            score = clip(
                student_ability
                - (difficulty - 1) * 5
                + rng.normal(0, 13),
                0,
                100,
            )

            attempts = int(
                clip(
                    rng.poisson(
                        1.2 + (100 - score) / 35
                    ) + 1,
                    1,
                    10,
                )
            )

            study_minutes = int(
                clip(
                    rng.normal(
                        25 + difficulty * 12
                        + (100 - score) * 0.35,
                        15,
                    ),
                    5,
                    240,
                )
            )

            if score >= 75:
                mastery = "Strong"
            elif score >= 50:
                mastery = "Moderate"
            else:
                mastery = "Weak"

            records.append({
                "student_id": student.student_id,
                "topic": topic,
                "topic_difficulty": difficulty,
                "topic_score": round(float(score), 1),
                "attempt_count": attempts,
                "study_minutes": study_minutes,
                "mastery_level": mastery,
            })

    return pd.DataFrame(records)


# ============================================================
# DATA CORRUPTION: INTRODUCE MODERATE MESSINESS
# ============================================================

def make_messy_students(clean_df):
    df = clean_df.copy()

    # Missing values in selected input features.
    df = add_missing_values(
        df,
        [
            "age",
            "previous_marks",
            "attendance_pct",
            "study_hours_per_week",
            "study_consistency_pct",
            "quiz_average",
            "assignment_average",
            "sleep_hours_per_night",
        ],
        rate=0.025,
    )

    # Slightly inconsistent category labels.
    for column in [
        "grade_level",
        "study_frequency",
        "learning_style",
        "internet_access",
        "performance_category",
        "risk_level",
    ]:
        df[column] = mess_up_categories(
            df[column],
            probability=0.12,
        )

    # A small number of invalid numeric entries/outliers.
    numeric_corruptions = {
        "attendance_pct": [-5, 108],
        "study_hours_per_week": [-2, 65],
        "quiz_average": [-8, 115],
        "assignment_average": [-4, 110],
    }

    for column, possible_values in numeric_corruptions.items():
        indices = rng.choice(
            df.index,
            size=max(1, int(len(df) * 0.005)),
            replace=False,
        )

        for index in indices:
            df.loc[index, column] = rng.choice(
                possible_values
            )

    # Duplicate a small number of complete records.
    duplicate_count = max(1, int(len(df) * 0.01))
    duplicate_rows = df.sample(
        n=duplicate_count,
        random_state=RANDOM_SEED,
    )

    df = pd.concat(
        [df, duplicate_rows],
        ignore_index=True,
    )

    # Shuffle so duplicates are not all at the end.
    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return df


def make_messy_topics(clean_topics):
    df = clean_topics.copy()

    df = add_missing_values(
        df,
        ["topic_score", "study_minutes"],
        rate=0.02,
    )

    df["topic"] = df["topic"].astype(object)

    topic_mask = rng.random(len(df)) < 0.04
    df.loc[topic_mask, "topic"] = (
        df.loc[topic_mask, "topic"].astype(str).str.lower()
    )

    # A few missing topic records can also be detected during EDA.
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic LearnIQ student records...")

    students_clean = generate_students()
    topics_clean = generate_topic_attempts(students_clean)

    students_messy = make_messy_students(students_clean)
    topics_messy = make_messy_topics(topics_clean)

    clean_path = (
        OUTPUT_DIR / "student_performance_clean_reference.csv"
    )
    student_path = (
        OUTPUT_DIR / "student_performance_messy.csv"
    )
    topic_path = (
        OUTPUT_DIR / "topic_attempts_messy.csv"
    )

    students_clean.to_csv(clean_path, index=False)
    students_messy.to_csv(student_path, index=False)
    topics_messy.to_csv(topic_path, index=False)

    print("\nDataset generation complete!")
    print(f"\nClean student records: {len(students_clean)}")
    print(f"Messy student records: {len(students_messy)}")
    print(f"Topic attempt records: {len(topics_messy)}")

    print("\nSaved files:")
    print(clean_path)
    print(student_path)
    print(topic_path)

    print("\nMissing values in messy student data:")
    print(
        students_messy.isna().sum()
        .loc[lambda x: x > 0]
        .to_string()
    )

    print("\nSample student records:")
    print(students_messy.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
