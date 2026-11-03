
import streamlit as st
import pandas as pd
import random

# ----------------------------------
# 1. Streamlit Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="TV Scheduling GA",
    page_icon="📺",
    layout="wide"
)

st.title("📺 TV Scheduling Using Genetic Algorithm")

st.write(
    "This application uses a Genetic Algorithm "
    "to generate an optimized TV schedule based "
    "on program ratings."
)

# ----------------------------------
# 2. Load Modified Dataset
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("modified_program_ratings.csv")

df = load_data()

ratings = {
    row["Type of Program"]: row.iloc[1:].tolist()
    for _, row in df.iterrows()
}

all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))

# ----------------------------------
# 3. Genetic Algorithm Functions
# ----------------------------------

def fitness_function(schedule):
    total_rating = 0

    for i, program in enumerate(schedule):
        total_rating += ratings[program][i]

    return total_rating


def initialize_population(population_size):
    return [
        [
            random.choice(all_programs)
            for _ in all_time_slots
        ]
        for _ in range(population_size)
    ]


def selection(population):
    candidates = random.sample(population, 3)

    return max(
        candidates,
        key=fitness_function
    ).copy()


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def mutate(schedule):
    new_schedule = schedule.copy()

    point = random.randint(
        0, len(new_schedule) - 1
    )

    new_schedule[point] = random.choice(
        all_programs
    )

    return new_schedule


def genetic_algorithm(
    generations,
    population_size,
    crossover_rate,
    mutation_rate,
    elitism_size=2
):

    population = initialize_population(population_size)

    for generation in range(generations):

        population.sort(
            key=fitness_function,
            reverse=True
        )

        new_population = [
            individual.copy()
            for individual in population[:elitism_size]
        ]

        while len(new_population) < population_size:

            parent1 = selection(population)
            parent2 = selection(population)

            if random.random() < crossover_rate:
                child1, child2 = crossover(
                    parent1, parent2
                )
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)

            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    best_schedule = max(
        population,
        key=fitness_function
    )

    best_fitness = fitness_function(best_schedule)

    return best_schedule, best_fitness


# ----------------------------------
# 4. Streamlit User Interface
# ----------------------------------

st.header("⚙️ Genetic Algorithm Parameters")

col1, col2 = st.columns(2)

with col1:
    CO_R = st.slider(
        "Crossover Rate (CO_R)",
        min_value=0.0,
        max_value=0.95,
        value=0.80,
        step=0.01
    )

with col2:
    MUT_R = st.slider(
        "Mutation Rate (MUT_R)",
        min_value=0.01,
        max_value=0.05,
        value=0.02,
        step=0.01
    )

GEN = 100
POP = 50

st.write("Generations:", GEN)
st.write("Population Size:", POP)

# ----------------------------------
# 5. Run Genetic Algorithm
# ----------------------------------

if st.button(
    "🚀 Run Genetic Algorithm",
    type="primary"
):

    best_schedule, best_fitness = genetic_algorithm(
        generations=GEN,
        population_size=POP,
        crossover_rate=CO_R,
        mutation_rate=MUT_R
    )

    schedule_df = pd.DataFrame({
        "Time Slot": [
            f"{hour:02d}:00"
            for hour in all_time_slots
        ],
        "Program": best_schedule,
        "Rating": [
            ratings[program][i]
            for i, program in enumerate(best_schedule)
        ]
    })

    st.success(
        "Genetic Algorithm completed successfully!"
    )

    st.subheader("📋 Generated TV Schedule")

    st.dataframe(
        schedule_df,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "Total Fitness Score",
        round(best_fitness, 2)
    )

    st.write("**Parameters Used:**")
    st.write("Crossover Rate:", CO_R)
    st.write("Mutation Rate:", MUT_R)

    # Download generated schedule
    csv = schedule_df.to_csv(index=False)

    st.download_button(
        label="Download TV Schedule",
        data=csv,
        file_name="generated_tv_schedule.csv",
        mime="text/csv"
    )
