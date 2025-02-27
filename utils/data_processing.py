import pandas as pd
import numpy as np

##TODO: Add function to resample full tasks for fixed time steps
##TODO: Add function to padd for fixed number of observations. In those, outside the screen should be identified with the same placeholder as the padding. 

class EyeTrackingProcessor:
    # def __init__(self):
        
    def read_tsv(self, path: str) -> pd.DataFrame:
        """Read a tsv file and check for mislabeling of tasks

        Args:
            path (str): Path to the tsv file

        Returns:
            pd.DataFrame: Dataframe with the data from the file
        """
        df = pd.read_csv(path, sep='\t')
        expected_tasks = [f"Task {i}" for i in range(1, 7)] + [f"Task {i} end" for i in range(1, 7)]
        task_counts = df['Event'].value_counts()
        
        for task in expected_tasks:
            if task not in task_counts or task_counts[task] != 6:
                print(f"Mislabeling detected for file {path}: '{task}' has {task_counts.get(task, 0)} occurrences instead of 6.")
        
        return df
    
    def load_data(self, paths: list[str]) -> list[pd.DataFrame]:
        """Load data from a list of paths

        Args:
            paths (list[str]): List of paths to the data files

        Returns:
            list[pd.DataFrame]: List of dataframes with the data from the files
        """
        all_files = []
        for i in range(len(paths)):
            file = self.read_tsv(paths[i])
            all_files.append(file)
        return all_files

    def task_range_finder(self, db: pd.DataFrame, tasks_arr: list[str]) -> dict[str, list[tuple[int, int]]]:
        """Find the start and end times of tasks in a dataframe

        Args:
            db (pd.DataFrame): Dataframe with the events
            tasks_arr (list[str]): List of tasks to extract start and end times

        Returns:
            dict[str, list[tuple[int, int]]]: Dictionary with task names as keys and a list of tuples with start and end times as values
        """
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
        """Extract features from a list of dataframes for a list of tasks and agregating them into a single dataframe

        Args:
            dfs (list[pd.DataFrame]): List of dataframes for each task
            tasks (list[str]): List of tasks to extract features from
            features (list[str]): List of features to extract

        Returns:
            pd.DataFrame: Dataframe with features extracted from the list of tasks
        """
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
        return pd.concat(full_dataset).reset_index(drop=True) #Reset index is important otherwise we can have several times the same index and it messed up the chunking

    def get_chunks(self, df: pd.DataFrame, chunk_duration: int, time_offset_chunk: int) -> pd.DataFrame:
        """Chunk a task into fixed time durations

        Args:
            df (pd.DataFrame): Dataframe output by get_features
            chunk_duration (int): Duration of each chunk in microseconds
            time_offset_chunk (int): Time offset between chunks in microseconds

        Returns:
            pd.DataFrame: Dataframe with tasks chunked into fixed time durations identified by column "chunk_id"
        """
        chunked_data = []  # Store individual chunk DataFrames

        for (participant, task, period), group in df.groupby(["Participant name", "Task_id", "Task_execution"]):
            start_time = group["Recording timestamp"].min()
            end_time = group["Recording timestamp"].max()
            chunk_start_times = np.arange(start_time, end_time - chunk_duration + 1, time_offset_chunk)
            
            chunk_id = 0
            for chunk_start in chunk_start_times:
                chunk_end = chunk_start + chunk_duration
                mask = (group["Recording timestamp"] >= chunk_start) & (group["Recording timestamp"] < chunk_end)
                chunk_subset = chunk_subset = group.loc[mask].copy() #Copying as a single timestamp is in several chunks
                
                # Ensure all chunks are the right duration (some full tasks might be shorter than the chunk duration)
                # if not chunk_subset.empty and (chunk_subset["Recording timestamp"].max() - chunk_start) >= self.chunk_duration:
                chunk_subset["chunk_id"] = chunk_id
                chunked_data.append(chunk_subset) 
                chunk_id += 1
                    

        df = pd.concat(chunked_data).reset_index(drop=True)
        df["chunk_id"] = df["chunk_id"].astype("Int64")
        return df

    def resample_chunks(self, df: pd.DataFrame, interpolate_col: list[str], target_size: int, chunk_col="chunk_id") -> pd.DataFrame:
        """Resample chunks to a fixed size

        Args:
            df (pd.DataFrame): dataframe of a task that has been chunked to fixed time durations
            interpolate_col (list[str]): columns to interpolate
            target_size (int): number of points per chunk
            chunk_col (str, optional): Column identifying chunks. Defaults to "chunk_id".

        Returns:
            pd.DataFrame: Dataframe of a task with fixed number of points per chunk
        """
        resampled_dfs = []

        for chunk_id, sub_df in df.groupby(chunk_col):
            sub_df = sub_df.sort_index()
            original_indices = np.linspace(0, 1, len(sub_df))
            new_indices = np.linspace(0, 1, target_size)

            resampled_sub_df = pd.DataFrame(index=new_indices)
            for col in interpolate_col:
                if col != chunk_col:
                    resampled_sub_df[col] = np.interp(new_indices, original_indices, sub_df[col])
            
            if resampled_sub_df.isna().all().any():
                print(f"Only Nans for participant {sub_df['Participant name'].iloc[0]}, task {sub_df['Task_id'].iloc[0]}, execution {sub_df['Task_execution'].iloc[0]}, chunk {chunk_id}")
                print(resampled_sub_df.columns[resampled_sub_df.isna().all()].tolist(), end="\n\n")
                continue

            resampled_sub_df[chunk_col] = chunk_id
            resampled_dfs.append(resampled_sub_df)
        if len(resampled_dfs) > 0:
            return pd.concat(resampled_dfs).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def fixed_length_resample(self, df: pd.DataFrame, interpolate_col: list[str], target_size: int, chunk_col="chunk_id") -> pd.DataFrame:
        """Apply resample_chunks to all tasks in a dataframe

        Args:
            df (pd.DataFrame): Dataframe ouput by get_chunks
            interpolate_col (list[str]): columns to interpolate
            target_size (int): number of points per chunk
            chunk_col (str, optional): Column identifying chunks. Defaults to "chunk_id".

        Returns:
            pd.DataFrame: Dataframe with all tasks resampled to fixed number of points per chunk
        """
        all_resampled_dfs = []
        unique_combinations = df[['Participant name', 'Task_id', 'Task_execution']].drop_duplicates()

        for _, row in unique_combinations.iterrows():
            subset = df.query(f"`Participant name` == {row['Participant name']} and Task_id == {row['Task_id']} and Task_execution == {row['Task_execution']}")
            if len(subset) > 0:
                resampled_subset = self.resample_chunks(subset, interpolate_col, target_size, chunk_col=chunk_col)
                resampled_subset['Participant name'] = row['Participant name']
                resampled_subset['Task_id'] = row['Task_id']
                resampled_subset['Task_execution'] = row['Task_execution']
                all_resampled_dfs.append(resampled_subset)

        return pd.concat(all_resampled_dfs).reset_index(drop=True)
    
    def outside_screen_placeholder(self, df: pd.DataFrame, x_lim: list[int], y_lim: list[int], placeholder_value = np.nan) -> pd.DataFrame:
        """Replace values outside the screen boundaries with a placeholder value

        Args:
            df (pd.DataFrame): dataframe with gaze and mouse positions
            x_lim (list[int]): x-axis limits of the screen (in pixels)
            y_lim (list[int]): y-axis limits of the screen (in pixels)
            placeholder_value (int, optional): placeholder value. Defaults to Nan.

        Returns:
            pd.DataFrame: Dataframe with values outside the screen replaced by the placeholder value
        """
        
        mask_gaze_x = (df['Gaze point X'] < x_lim[0]) | (df['Gaze point X'] > x_lim[1])
        mask_gaze_y = (df['Gaze point Y'] < y_lim[0]) | (df['Gaze point Y'] > y_lim[1])
        df.loc[mask_gaze_x | mask_gaze_y, ['Gaze point X', 'Gaze point Y']] = placeholder_value
        
        mask_mouse_x = (df['Mouse position X'] < x_lim[0]) | (df['Mouse position X'] > x_lim[1])
        mask_mouse_y = (df['Mouse position Y'] < y_lim[0]) | (df['Mouse position Y'] > y_lim[1])
        df.loc[mask_mouse_x | mask_mouse_y, ['Mouse position X', 'Mouse position Y']] = placeholder_value

        return df
    
    def resample_task(self, df: pd.DataFrame, interpolate_col: list[str], timestep) -> pd.DataFrame:
        """Resample a task to fixed time steps

        Args:
            df (pd.DataFrame): dataframe built by the get_features method
            interpolate_col (list[str]): features to interpolate
            timestep (int, optional): resampling time step in seconds

        Returns:
            pd.DataFrame: task resampled to fixed time increments
        """
        
        timestep = timestep * 1e6 # Convert to microseconds
        min_time = df['Recording timestamp'].min()
        max_time = df['Recording timestamp'].max()
        new_timestamps = np.arange(min_time, max_time, timestep)
        resampled_df = pd.DataFrame(index=new_timestamps)
        
        for col in interpolate_col:
            if col != 'Recording timestamp':
                resampled_df[col] = np.interp(new_timestamps, df['Recording timestamp'], df[col])
        
        return resampled_df.reset_index().rename(columns={'index': 'Recording timestamp'})
    
    def fixed_time_steps_resample(self, df: pd.DataFrame, interpolate_col: list[str], timestep: int=0.001, pad_value= np.nan) -> list[pd.DataFrame]:
        """Resample all tasks to fixed time steps and pad with placeholder values to match the longest task

        Args:
            df (pd.DataFrame): dataframe built by the get_features method
            interpolate_col (list[str]): features to interpolate
            timestep (int, optional): resampling time step in seconds: Defaults to 0.001.

        Returns:
            pd.DataFrame: all tasks resampled to fixed time increments
        """
        
        all_resampled_dfs = []
        unique_combinations = df[['Participant name', 'Task_id', 'Task_execution']].drop_duplicates()
        max_len = 0 #initialize max length tracker

        resampled_tasks = []
        for _, row in unique_combinations.iterrows():
            subset = df.query(f"`Participant name` == {row['Participant name']} and Task_id == {row['Task_id']} and Task_execution == {row['Task_execution']}")
            if len(subset) > 0:
                resampled_task = self.resample_task(subset, interpolate_col, timestep)
                resampled_task['Participant name'] = row['Participant name']
                resampled_task['Task_id'] = row['Task_id']
                resampled_task['Task_execution'] = row['Task_execution']
                resampled_tasks.append(resampled_task)
                max_len = max(max_len, len(resampled_task))
        
        for task_df in resampled_tasks:
            current_len = len(task_df)
            if current_len < max_len:
                padding_df = pd.DataFrame({col: pad_value for col in task_df.columns}, index=range(current_len, max_len))
                padded_task = pd.concat([task_df, padding_df], ignore_index=True)
                padded_task['Participant name'] = task_df['Participant name'].iloc[0]
                padded_task['Task_id'] = task_df['Task_id'].iloc[0]
                padded_task['Task_execution'] = task_df['Task_execution'].iloc[0]
            else:
                padded_task = task_df
            all_resampled_dfs.append(padded_task)

        return pd.concat(all_resampled_dfs, ignore_index=True).reset_index(drop=True)
            
