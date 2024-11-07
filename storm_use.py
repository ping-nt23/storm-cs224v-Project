import os
from argparse import ArgumentParser
from knowledge_storm.rm import BingSearch
from knowledge_storm.lm import TogetherClient
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs

def main(args):
    lm_configs = STORMWikiLMConfigs()
    together_kwargs = {
        'api_key': os.getenv("TOGETHER_API_KEY"),
        'temperature': 1.0,
        'top_p': 0.9,
        "stop": ('\n\n---',)
    }
    # STORM is a LM system so different components can be powered by different models to reach a good balance between cost and quality.
    # For a good practice, choose a cheaper/faster model for `conv_simulator_lm` which is used to split queries, synthesize answers in the conversation.
    # Choose a more powerful model for `article_gen_lm` to generate verifiable text with citations.
    llama_8B = TogetherClient(model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', max_tokens=500, **together_kwargs)
    # llama_70B = TogetherClient(model='gpt-4o', max_tokens=3000, **together_kwargs)
    lm_configs.set_conv_simulator_lm(llama_8B)
    lm_configs.set_question_asker_lm(llama_8B)
    lm_configs.set_outline_gen_lm(llama_8B)
    lm_configs.set_article_gen_lm(llama_8B)
    lm_configs.set_article_polish_lm(llama_8B)
    # # Check out the STORMWikiRunnerArguments class for more configurations.
    engine_args = STORMWikiRunnerArguments(
        # max_search_queries=args.max_search_queries,
        # max_search_thread=args.max_search_thread,
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )
    rm = BingSearch(bing_search_api_key='') # replace with Bing API key
    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    topic = input('Topic: ')
    # topic = 'Recent News about' + input('Topic: ')
    try:
        runner.run(
            topic=topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
            remove_duplicate=False
        )
        runner.post_run()
        runner.summary()
    except Exception as e:
        raise

if __name__ == '__main__':
    parser = ArgumentParser()
    # global arguments
    parser.add_argument('--output-dir', type=str, default='./results/gpt',
                        help='Directory to store the outputs.')
    parser.add_argument('--max-thread-num', type=int, default=1,
                        help='Maximum number of threads to use. The information seeking part and the article generation'
                             'part can speed up by using multiple threads. Consider reducing it if keep getting '
                             '"Exceed rate limit" error when calling LM API.')
    # stage of the pipeline
    parser.add_argument('--do-research', action='store_true',
                        help='If True, simulate conversation to research the topic; otherwise, load the results.')
    parser.add_argument('--do-generate-outline', action='store_true',
                        help='If True, generate an outline for the topic; otherwise, load the results.')
    parser.add_argument('--do-generate-article', action='store_true',
                        help='If True, generate an article for the topic; otherwise, load the results.')
    parser.add_argument('--do-polish-article', action='store_true',
                        help='If True, polish the article by adding a summarization section and (optionally) removing '
                             'duplicate content.')
    # hyperparameters for the pre-writing stage
    parser.add_argument('--max-conv-turn', type=int, default=2,
                        help='Maximum number of questions in conversational question asking.')
    parser.add_argument('--max-perspective', type=int, default=2,
                        help='Maximum number of perspectives to consider in perspective-guided question asking.')
    parser.add_argument('--search-top-k', type=int, default=3,
                        help='Top k search results to consider for each search query.')
    # hyperparameters for the writing stage
    parser.add_argument('--retrieve-top-k', type=int, default=3,
                        help='Top k collected references for each section title.')
    parser.add_argument('--remove-duplicate', action='store_true',
                        help='If True, remove duplicate content from the article.')
    
    # parser.add_argument(
    #     '--max_search_queries',
    #     type=int,
    #     default=1,
    #     help='Maximum number of search queries to consider for each question.'
    # )
    main(parser.parse_args())

# TogetherClient(dspy.HFModel):
#     """A wrapper class for dspy.Together."""

#     def __init__(
#         self,
#         model,
#         api_key: Optional[str] = None,
#         apply_tokenizer_chat_template=False,
#         hf_tokenizer_name=None,
#         model_type: Literal["chat", "text"] = "chat",
#         **kwargs,
#     ):