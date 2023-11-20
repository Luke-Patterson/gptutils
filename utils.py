import pandas as pd
import math
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import tiktoken
import openai
import io

class Conversation:
    def __init__(self):
        self.user_messages = []
        self.assistant_messages = []

    def ingest_openai_response(self, messages: list):
        for m in messages:
            if m['role'] == 'user':
                self.user_messages.append(m['content'])
            if m['role'] == 'assistant':
                self.assistant_messages.append(m['content'])

    def print_conversation(self, last_n_messages = None):
        '''
        print conversation to console.
        params:
        last_n_messages - only return last n messages in the conversation
        '''
        assert len(self.user_messages) == len(self.assistant_messages), 'user and assistant messages not same length'
        for n in range(len(self.user_messages)):
            if last_n_messages is None or n + last_n_messages >= len(self.user_messages):
                print('User:', self.user_messages[n])
                print('AI:', self.assistant_messages[n])

    def return_conversation(self):
        '''
        returns conversation as a string
        '''
        convo_string = ''
        assert len(self.user_messages) == len(self.assistant_messages), 'user and assistant messages not same length'
        for n in range(len(self.user_messages)):
            convo_string += 'User:'
            convo_string += self.user_messages[n]
            convo_string += '\n'
            convo_string += 'AI:'
            convo_string += self.assistant_messages[n]
            convo_string += '\n'
        return convo_string

    def return_response(self):
        if len(self.assistant_messages) == 1:
            return self.assistant_messages[0]
        else:
            return self.assistant_messages


def getKey():
    # The URL to Azure Key Vault
    vault_url = 'https://openai-poc-techsolutions.vault.azure.net/'

    # The Azure Key Vault Secret Name for the OpenAI API Key
    secret_name = 'azureopenai-apikey'
    # Log on to the key vault using the Azure Credential
    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)

    secret_client = SecretClient(vault_url=vault_url, credential=credential)

    # Get the secret from the Key Vault, set the API key used by the openai module
    secret = secret_client.get_secret(secret_name)
    return secret.value


def num_tokens_from_string(string) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_chatgpt_response(prompts: list, sys_context=None ,verbose = False, engine = 'gpt-35-turbo-16k', frontload_prompts = False) -> Conversation:
    if verbose:
        print('starting chat with', len(prompts), 'messages')
    token_count_used = 0
    messages = []
    if sys_context is not None:
        messages.append({"role": "system", "content": sys_context})
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        for msg in messages:
            token_count_used += num_tokens_from_string(msg['content'])
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=messages)
        text = response['choices'][0]['message']['content']
        token_count_used += num_tokens_from_string(text)
        messages.append({'role': 'assistant', "content": text})
    if verbose:
        print('Tokens used:', token_count_used)
    conversation = Conversation()
    conversation.ingest_openai_response(messages)
    return conversation

def chatgpt_analyze_table(df : pd.DataFrame, question : str, context_window = 16385):
    '''
    Have chatgpt analyze a table.
    First, Parse a table into CSV input format for ChatGPT. If the table exceeds the context window, break it up into chunks.
    Then, feed the table into chatgpt along with the prompt

    params:
    df - dataframe to feed to chatgpt
    context_window - max context window to work with. Defaults to gpt-35-turbo-16k max context window.
    prompt - question to ask chatgpt about the data

    returns:
    string of conversation
    '''


    # stream the pandas dataframe as a csv
    stream = io.StringIO()
    df.to_csv(stream, sep = ',')


    # get token length of table
    table_tokens = num_tokens_from_string(stream.getvalue())

    # subtract the context window used by the rest of the prompt
    context = 'You are skilled data analyst preparing to analyze some tables.'
    prompt = 'I am going to feed you a table in CSV format and then ask you to answer questions about the table. Do not '+\
    ' return code on how to solve it, we want to give the answer to the question. Here is the table:'
    adj_window = context_window - num_tokens_from_string(context) + num_tokens_from_string(prompt)

    if table_tokens > adj_window:
        raise('table exceeds context window')

        # tried a workaround splitting the table into multiple messages, doesn't work

        # initial_prompt = 'I am going to feed you a table in CSV format over the next several messages. I will then ask' +\
        #     ' you questions about the table.'
        #
        # inter_table_prompt = 'Here is the next chunk of the table. The column headers remain the same. '
        # inter_tokens = num_tokens_from_string(inter_table_prompt)
        # adj_window2 = context_window - inter_tokens
        #
        # csv_lines = stream.getvalue().split('\n')
        # prompts = [initial_prompt]
        # print('processing ', len(csv_lines), 'records')
        # working_prompt = inter_table_prompt
        # working_tokens = inter_tokens
        # for n, line in enumerate(csv_lines):
        #     if n % 100 == 0:
        #         print('Line',n)
        #
        #     line_tokens = num_tokens_from_string(line) + 1 # add 1 because we removed the line break
        #     if working_tokens + line_tokens < adj_window2:
        #         working_prompt = working_prompt + line +'\n'
        #         working_tokens += line_tokens
        #     else:
        #         prompts.append(working_prompt)
        #         working_prompt = inter_table_prompt + line +'\n'
        #         working_tokens = inter_tokens + line_tokens
        # prompts.append(working_prompt)
        #
        # final_prompt = ' That is the end of the table. Now I am going to ask you some questions about the table: ' + questions
        # prompts.append(final_prompt)


    else:


        prompts = [prompt + stream.getvalue(), question]


    conversation = get_chatgpt_response(prompts=prompts, sys_context=context)
    return conversation

def setup_langchain_AzureOpenAI(deployment = "text-davinci-003",model_name = "text-davinci-003", api_version = '2022-12-01'):
    openai.api_key = getKey()
    os.environ["OPENAI_API_VERSION"] = api_version
    llm = AzureOpenAI(openai_api_key=openai.api_key, deployment_name= deployment,
                      model_name=model_name)
    return llm
