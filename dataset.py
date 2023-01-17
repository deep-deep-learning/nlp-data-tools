import pandas as pd
from transformers import GPT2TokenizerFast

class Dataset:

    def __init__(
        self,
        file_path:str,
    ):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.origin_df = self.df.copy()

    def prepare_dataset_openai(
        self,
    ):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.prepare_dataset(
            completion_prefix=' ',
            completion_suffix=' END',
            prompt_suffix='\n\n###\n\n'
        )
    
    def prepare_dataset(
        self,
        prompt_tokens:int=1800,
        completion_prefix:str=None,
        completion_suffix:str=None,
        prompt_prefix:str=None,
        prompt_suffix:str=None,
    ):

        # check if required columns are there
        assert 'completion' in self.df.columns, "We need a column named 'completion'"
        assert 'prompt' in self.df.columns, "We need a column named 'prompt'"

        # get first 1800 tokens of prompts
        self.df['prompt'] = self.df['prompt'].apply(lambda c: self.get_first_n_tokens(c, prompt_tokens))

        print('Formatting completions')
        # prepare completion
        if completion_prefix != None:
            self.check_and_add_fix(self.df, 'completion', completion_prefix, 'prefix')
        if completion_suffix != None:
            self.check_and_add_fix(self.df, 'completion', completion_suffix, 'suffix')

        print('Formatting prompts')
        # prepare prompt
        if prompt_prefix != None:
            self.check_and_add_fix(self.df, 'prompt', prompt_prefix, 'prefix')
        if prompt_suffix != None:
            self.check_and_add_fix(self.df, 'prompt', prompt_suffix, 'suffix')

        output_path = self.file_path[:-4] + '-formatted.csv'
        print('Saving formatted dataset', output_path)
        self.df[['prompt', 'completion']].to_csv(output_path, index=False)

    def get_first_n_tokens(
        self,
        text:str,
        n:int,
    ) -> str:
        
        return self.tokenizer.decode(self.tokenizer.encode(text)[:n])
        
    def check_and_add_fix(
        self,
        column:str,
        fix:str,
        option:str,
    ):
        
        # get indices with no prefix/suffix
        if option == 'prefix':
            no_fix = self.df[column].str[:len(fix)] != fix
        elif option == 'suffix':
            no_fix = self.df[column].str[-len(fix):] != fix
        no_fix_index = self.df[no_fix].index.to_list()
        print(f'The indices of the samples that do not have the {option}: {no_fix_index}')

        if len(no_fix_index) > 0:
            # add prefix/suffix
            print(f'Adding the {option} to them')
            if option == 'prefix':
                self.df.loc[no_fix, column] = self.df.loc[no_fix, column].apply(lambda c: f'{fix}{c}')
            elif option == 'suffix':
                self.df.loc[no_fix, column] = self.df.loc[no_fix, column].apply(lambda c: f'{c}{fix}')