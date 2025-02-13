import pandas as pd
import numpy as np

class EyeTrackingProcessor:
    def __init__(self, chunk_size, min_duration, target_size):
        self.chunk_size = chunk_size
        self.min_duration = min_duration
        self.target_size = target_size

    def read_tsv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t')
        expected_tasks = [f"Task {i}" for i in range(1, 7)] + [f"Task {i} end" for i in range(1, 7)]
        task_counts = df['Event'].value_counts()
        
        for task in expected_tasks:
            if task not in task_counts or task_counts[task] != 6:
                print(f"Mislabeling detected for file {path}: '{task}' has {task_counts.get(task, 0)} occurrences instead of 6.")
        
        return df
    
    def load_data(self, paths: list[str]) -> list[pd.DataFrame]:
        all_files = []
        for i in range(len(paths)):
            file = self.read_tsv(paths[i])
            all_files.append(file)
        return all_files

    def task_range_finder(self, db: pd.DataFrame, tasks_arr: list[str]) -> dict[str, list[tuple[int, int]]]:
        event_df = db.loc[db['Event'].str.contains('Task', na=False)][["Event","Recording timestamp"]].sort_values(by="Recording timestamp").reset_index(drop=True)
        task_ranges = {task: [] for task in tasks_arr}
        task_stack = {}

        for _, row in event_df.iterrows():
            event, timestamp = row["Event"], row["Recording timestamp"]
            if "Task" in event and "end" not in event:
                task_stack[event] = timestamp  # Store start time
            elif "end" in event:
                task_type = event.replace(" end", "")
                if task_type in task_stack:
                    task_ranges[task_type].append((task_stack.pop(task_type), timestamp))
        
        return task_ranges

    def get_features(self, dfs: list[pd.DataFrame], tasks: list[str], features: list[str]) -> pd.DataFrame:
        full_dataset = []
        for df in dfs:
            sub_dataset = []
            tasks_ranges = self.task_range_finder(df, tasks)
            for task in tasks_ranges:
                for i, period in enumerate(tasks_ranges[task]):
                    task_data = df.query(f"`Recording timestamp` >= {period[0]} and `Recording timestamp` <= {period[1]}")[features]
                    task_data = task_data.ffill().bfill()
                    task_data["Task_id"] = int(task[-1])
                    task_data["Task_execution"] = i
                    sub_dataset.append(task_data)
            full_dataset.append(pd.concat(sub_dataset))
        return pd.concat(full_dataset)

    def get_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        df["chunk_id"] = np.nan

        for (participant, task, period), group in df.groupby(["Participant name", "Task_id", "Task_execution"]):
            start_time = group["Recording timestamp"].iloc[0]
            chunk_ids = (group["Recording timestamp"] - start_time) // self.chunk_size
            if (group["Recording timestamp"].iloc[-1] - start_time) % self.chunk_size < self.min_duration:
                chunk_ids[chunk_ids == chunk_ids.max()] = np.nan
            df.loc[group.index, "chunk_id"] = chunk_ids

        df["chunk_id"] = df["chunk_id"].astype("Int64")
        return df

    def resample_chunks(self, df: pd.DataFrame, interpolate_col: list[str], chunk_col="chunk_id") -> pd.DataFrame:
        resampled_dfs = []

        for chunk_id, sub_df in df.groupby(chunk_col):
            sub_df = sub_df.sort_index()
            original_indices = np.linspace(0, 1, len(sub_df))
            new_indices = np.linspace(0, 1, self.target_size)

            resampled_sub_df = pd.DataFrame(index=new_indices)
            for col in interpolate_col:
                if col != chunk_col:
                    resampled_sub_df[col] = np.interp(new_indices, original_indices, sub_df[col])
            
            if resampled_sub_df.isna().all().any():
                print(f"Processing participant {sub_df['Participant name'].iloc[0]}, task {sub_df['Task_id'].iloc[0]}, execution {sub_df['Task_execution'].iloc[0]}, chunk {chunk_id}")
                print(resampled_sub_df.columns[resampled_sub_df.isna().all()].tolist(), end="\n\n")

            resampled_sub_df[chunk_col] = chunk_id
            resampled_dfs.append(resampled_sub_df)

        return pd.concat(resampled_dfs).reset_index(drop=True)

    def process_and_resample_all(self, df: pd.DataFrame, interpolate_col: list[str], chunk_col="chunk_id") -> pd.DataFrame:
        all_resampled_dfs = []
        unique_combinations = df[['Participant name', 'Task_id', 'Task_execution']].drop_duplicates()

        for _, row in unique_combinations.iterrows():
            subset = df.query(f"`Participant name` == {row['Participant name']} and Task_id == {row['Task_id']} and Task_execution == {row['Task_execution']}")
            if len(subset) > 0:
                resampled_subset = self.resample_chunks(subset, interpolate_col, chunk_col=chunk_col)
                resampled_subset['Participant name'] = row['Participant name']
                resampled_subset['Task_id'] = row['Task_id']
                resampled_subset['Task_execution'] = row['Task_execution']
                all_resampled_dfs.append(resampled_subset)

        return pd.concat(all_resampled_dfs).reset_index(drop=True)
