Config structure:


1. model:
   	model_name:

2. data:
	task: 'name_of_task' - used as base for output file
	data_type: 'list' or None
	file_list: '.*jsonl'	

	data_loader: generic_hf_loader
   	data_loader_args: 
   		dataset_name
		* split
		* subset_size
		target_column

	data_loader: load_jsonl
	data_loader_args: 
		dataset_path
		* subset_size
		target_column
	
	system_prompt_type: 'list'
	system_prompt_path: directory, with corresponding filename shared with file_list	
