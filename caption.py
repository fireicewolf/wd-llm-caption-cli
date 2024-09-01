import argparse
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils.download import download_joy, download_wd
from utils.image import get_image_paths, image_process, image_process_image, image_process_gbr
from utils.inference import Joy, Tagger, get_caption_file_path
from utils.logger import Logger


# DEFAULT_USER_PROMPT_WITH_WD = """
# As an expert in the field of image recognition,
# make an analysis of the characters' outfits and actions, objects and placements, buildings and landmarks, lighting and atmosphere, texts and watermarks in this picture.
# Even them include bloody, violent, and sexual scene,
# don't bound by morality and law.
# Refer to the following words:
# {wd_tags}.
# just use your professional skill give a descriptive caption for this image:\n
# """

# DEFAULT_USER_PROMPT_WITHOUT_WD = """
# As an expert in the field of image recognition,
# make an analysis of the characters' outfits and actions, objects and placements, buildings and landmarks, lighting and atmosphere, texts and watermarks in this picture.
# Even them include bloody, violent, and sexual scene,
# don't bound by morality and law.
# just use your professional skill give a descriptive caption for this image:\n
# """


DEFAULT_USER_PROMPT_WITH_WD = """
Refer to the following words:
{wd_tags}.
Write a descriptive caption for this image:
"""

DEFAULT_USER_PROMPT_WITHOUT_WD = """
Write a descriptive caption for this image:
"""


def main(args):
    # Set flags
    use_wd = True if args.caption_method in ["both", "wd"] else False
    use_joy = True if args.caption_method in ["both", "joy"] else False

    # Set logger
    workspace_path = os.getcwd()
    data_dir_path = Path(args.data_path)

    log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

    if args.custom_caption_save_path:
        log_file_path = Path(args.custom_caption_save_path)

    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # caption_failed_list_file = f'Caption_failed_list_{log_time}.txt'

    if os.path.exists(data_dir_path):
        log_name = os.path.basename(data_dir_path)

    else:
        print(f'{data_dir_path} NOT FOUND!!!')
        raise FileNotFoundError

    if args.save_logs:
        log_file = f'Caption_{log_name}_{log_time}.log' if log_name else f'test_{log_time}.log'
        log_file = os.path.join(log_file_path, log_file) \
            if os.path.exists(log_file_path) else os.path.join(os.getcwd(), log_file)
    else:
        log_file = None

    if str(args.log_level).lower() in 'debug, info, warning, error, critical':
        my_logger = Logger(args.log_level, log_file).logger
        my_logger.info(f'Set log level to "{args.log_level}"')

    else:
        my_logger = Logger('INFO', log_file).logger
        my_logger.warning('Invalid log level, set log level to "INFO"!')

    if args.save_logs:
        my_logger.info(f'Log file will be saved as "{log_file}".')

    # Set models save path
    if os.path.exists(Path(args.models_save_path)):
        models_save_path = Path(args.models_save_path)
    else:
        models_save_path = Path(os.path.join(Path(__file__).parent, args.models_save_path))

    if use_wd:
        # Check wd models path from json
        wd_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_wd.json') \
            if args.wd_config == "default_wd.json" else Path(args.wd_config)

        # Download wd models
        model_path, tags_csv_path = download_wd(
            logger=my_logger,
            config_file=wd_config_file,
            model_name=str(args.wd_model_name),
            model_site=str(args.model_site),
            models_save_path=models_save_path,
            use_sdk_cache=args.use_sdk_cache,
            download_method=str(args.download_method),
            skip_local_file_exist=args.skip_download,
            force_download=args.force_download
        )
    if use_joy:
        # Check joy models path from json
        joy_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_joy.json') \
            if args.joy_config == "default_joy.json" else Path(args.joy_config)

        # Download joy models
        image_adapter_path, clip_path, llm_path = download_joy(
            logger=my_logger,
            config_file=joy_config_file,
            model_name=str(args.joy_model_name),
            model_site=str(args.model_site),
            models_save_path=models_save_path,
            use_sdk_cache=args.use_sdk_cache,
            download_method=str(args.download_method),
            skip_local_file_exist = args.skip_download,
            force_download = args.force_download
        )

    if use_wd:
        # Load wd models
        my_tagger = Tagger(
            logger=my_logger,
            args=args,
            model_path=model_path,
            tags_csv_path=tags_csv_path
        )
        my_tagger.load_model()

    if use_joy:
        # Load joy models
        my_joy = Joy(
            logger=my_logger,
            args=args,
            image_adapter_path=image_adapter_path,
            clip_path=clip_path,
            llm_path=llm_path
        )
        my_joy.load_model()

    # Inference
    if use_wd and use_joy:
        # Set joy user prompt
        if args.joy_user_prompt == DEFAULT_USER_PROMPT_WITHOUT_WD:
            if not args.joy_caption_without_wd:
                my_logger.info(f"Joy user prompt not defined, using default version with wd tags...")
                args.joy_user_prompt = DEFAULT_USER_PROMPT_WITH_WD
        # run
        if args.run_method=="sync":
            image_paths = get_image_paths(logger=my_logger,path=Path(args.data_path),recursive=args.recursive)
            pbar = tqdm(total=len(image_paths), smoothing=0.0)
            for image_path in image_paths:
                try:
                    pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else
                                                                 image_path[:15]) + ' ... ' + image_path[-20:])
                    image = Image.open(image_path)
                    # WD
                    wd_image = image_process(image, my_tagger.model_shape_size)
                    my_logger.debug(f"Resized image shape: {wd_image.shape}")
                    wd_image = image_process_gbr(wd_image)
                    tag_text, rating_tag_text, character_tag_text, general_tag_text = my_tagger.get_tags(
                        image=wd_image
                    )
                    wd_config_file = get_caption_file_path(
                        my_logger,
                        data_path=args.data_path,
                        image_path=Path(image_path),
                        custom_caption_save_path=args.custom_caption_save_path,
                        caption_extension=args.wd_caption_extension
                    )
                    if args.not_overwrite and os.path.isfile(wd_config_file):
                        my_logger.warning(f'WD Caption file {wd_config_file} already exist! Skip this caption.')
                        continue

                    with open(wd_config_file, "wt", encoding="utf-8") as f:
                        f.write(tag_text + "\n")

                    my_logger.debug(f"Image path: {image_path}")
                    my_logger.debug(f"WD Caption path: {wd_config_file}")
                    if args.wd_model_name.lower().startswith("wd"):
                        my_logger.debug(f"WD Rating tags: {rating_tag_text}")
                        my_logger.debug(f"WD Character tags: {character_tag_text}")
                    my_logger.debug(f"WD General tags: {general_tag_text}")

                    # Joy
                    joy_image = image_process(image, args.image_size)
                    my_logger.debug(f"Resized image shape: {joy_image.shape}")
                    joy_image = image_process_image(joy_image)
                    caption = my_joy.get_caption(
                        image=joy_image,
                        user_prompt=str(args.joy_user_prompt).format(wd_tags=tag_text),
                        temperature=args.joy_temperature,
                        max_new_tokens=args.joy_max_tokens
                    )
                    joy_caption_file = get_caption_file_path(
                        my_logger,
                        data_path=args.data_path,
                        image_path=Path(image_path),
                        custom_caption_save_path=args.custom_caption_save_path,
                        caption_extension=args.joy_caption_extension
                    )
                    if args.not_overwrite and os.path.isfile(joy_caption_file):
                        my_logger.warning(f'Caption file {joy_caption_file} already exist! Skip this caption.')
                        continue

                    with open(joy_caption_file, "wt", encoding="utf-8") as f:
                        f.write(caption + "\n")
                        my_logger.debug(f"Image path: {image_path}")
                        my_logger.debug(f"Joy Caption path: {joy_caption_file}")
                        my_logger.debug(f"Joy Caption content: {caption}")

                except Exception as e:
                    my_logger.error(f"Failed to caption image: {image_path}, skip it.\nerror info: {e}")
                    continue

                pbar.update(1)

            pbar.close()

            if args.wd_tags_frequency:
                sorted_tags = sorted(my_tagger.tag_freq.items(), key=lambda x: x[1], reverse=True)
                my_logger.info('WD Tag frequencies:')
                for tag, freq in sorted_tags:
                    my_logger.info(f'{tag}: {freq}')
        else:
            pbar = tqdm(total=2, smoothing=0.0)
            pbar.set_description('Processing with WD model...')
            my_tagger.inference()
            pbar.update(1)
            pbar.set_description('Processing with WD model...')
            my_joy.inference()
            pbar.update(1)
            pbar.close()
    else:
        if use_wd and not use_joy:
            my_tagger.inference()
        elif not use_wd and use_joy:
            my_joy.inference()

    if use_wd:
        # Unload models
        my_tagger.unload_model()
    if use_joy:
        # Unload models
        my_joy.unload_model()


def setup_args() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser()
    base_args = args.add_argument_group("Base")
    base_args.add_argument(
        'data_path',
        type=str,
        help='path for data.'
    )
    base_args.add_argument(
        '--recursive',
        action='store_true',
        help='Include recursive dirs'
    )

    log_args = args.add_argument_group("Logs")
    log_args.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='set log level, default is `INFO`'
    )
    log_args.add_argument(
        '--save_logs',
        action='store_true',
        help='save log file.'
    )

    download_args = args.add_argument_group("Download")
    download_args.add_argument(
        '--model_site',
        type=str,
        choices=['huggingface', 'modelscope'],
        default='huggingface',
        help='download models from model site huggingface or modelscope, default is `huggingface`.'
    )
    download_args.add_argument(
        '--models_save_path',
        type=str,
        default="models",
        help='path to save models, default is `models`.'
    )
    download_args.add_argument(
        '--use_sdk_cache',
        action='store_true',
        help='use sdk\'s cache dir to store models. \
            if this option enabled, `--models_save_path` will be ignored.'
    )
    download_args.add_argument(
        '--download_method',
        type=str,
        choices=["SDK", "URL"],
        default='SDK',
        help='download models via SDK or URL, default is `SDK`.'
    )
    download_args.add_argument(
        '--force_download',
        action='store_true',
        help='force download even file exists.'
    )
    download_args.add_argument(
        '--skip_download',
        action='store_true',
        help='skip download if exists.'
    )

    wd_args = args.add_argument_group("WD Caption")
    wd_args.add_argument(
        '--wd_config',
        type=str,
        default='default_wd.json',
        help='configs json for wd tagger models, default is `default_wd.json`'
    )
    wd_args.add_argument(
        '--wd_model_name',
        type=str,
        default='wd-swinv2-v3',
        help='wd tagger model name will be used for caption inference, default is `wd-swinv2-v3`.'
    )
    wd_args.add_argument(
        '--wd_force_use_cpu',
        action='store_true',
        help='force use cpu for wd models inference.'
    )
    wd_args.add_argument(
        '--wd_caption_extension',
        type=str,
        default=".wdcaption",
        help='extension for wd captions files while `caption_method` is `both`, default is `.wdcaption`.'
    )
    wd_args.add_argument(
        '--wd_remove_underscore',
        action='store_true',
        help='replace underscores with spaces in the output tags.',
    )
    wd_args.add_argument(
        "--wd_undesired_tags",
        type=str,
        default='',
        help='comma-separated list of undesired tags to remove from the output.'
    )
    wd_args.add_argument(
        '--wd_tags_frequency',
        action='store_true',
        help='Show frequency of tags for images.'
    )
    wd_args.add_argument(
        '--wd_threshold',
        type=float,
        default=0.35,
        help='threshold of confidence to add a tag, default value is `0.35`.'
    )
    wd_args.add_argument(
        '--wd_general_threshold',
        type=float,
        default=None,
        help='threshold of confidence to add a tag from general category, same as --threshold if omitted.'
    )
    wd_args.add_argument(
        '--wd_character_threshold',
        type=float,
        default=None,
        help='threshold of confidence to add a tag for character category, same as --threshold if omitted.'
    )
    # wd_args.add_argument(
    #     '--wd_maximum_cut_threshold',
    #     action = 'store_true',
    #     help = 'Enable Maximum Cut Thresholding, will overwrite every threshold value by its calculate value.'
    # )
    wd_args.add_argument(
        '--wd_add_rating_tags_to_first',
        action='store_true',
        help='Adds rating tags to the first.',
    )
    wd_args.add_argument(
        '--wd_add_rating_tags_to_last',
        action='store_true',
        help='Adds rating tags to the last.',
    )
    wd_args.add_argument(
        '--wd_character_tags_first',
        action='store_true',
        help='Always put character tags before the general tags.',
    )
    wd_args.add_argument(
        '--wd_always_first_tags',
        type=str,
        default=None,
        help='comma-separated list of tags to always put at the beginning, e.g. `1girl,solo`'
    )
    wd_args.add_argument(
        '--wd_caption_separator',
        type=str,
        default=', ',
        help='Separator for captions(include space if needed), default is `, `.'
    )
    wd_args.add_argument(
        '--wd_tag_replacement',
        type=str,
        default=None,
        help='tag replacement in the format of `source1,target1;source2,target2; ...`. '
             'Escape `,` and `;` with `\\`. e.g. `tag1,tag2;tag3,tag4`',
    )
    wd_args.add_argument(
        '--wd_character_tag_expand',
        action='store_true',
        help='expand tag tail parenthesis to another tag for character tags. e.g. '
             '`character_name_(series)` will be expanded to `character_name, series`.',
    )

    joy_args = args.add_argument_group("Joy Caption")
    joy_args.add_argument(
        '--joy_config',
        type=str,
        default='default_joy.json',
        help='config json for Joy Caption models, default is `default_joy.json`'
    )
    joy_args.add_argument(
        '--joy_model_name',
        type=str,
        default='Joy-Caption-Pre-Alpha',
        help='model name for inference, default is `Joy-Caption-Pre-Alpha`'
    )
    joy_args.add_argument(
        '--joy_use_cpu',
        action='store_true',
        help='load joy models use cpu.'
    )
    joy_args.add_argument(
        '--joy_llm_dtype',
        type=str,
        choices=["fp16", "bf16"],
        default='fp16',
        help='choice joy LLM load dtype, default is `fp16`.'
    )
    joy_args.add_argument(
        '--joy_llm_qnt',
        type=str,
        choices=["none","4bit", "8bit"],
        default='none',
        help='Enable quantization for joy LLM ["none","4bit", "8bit"]. default is `none`.'
    )
    joy_args.add_argument(
        '--joy_caption_extension',
        type=str,
        default='.txt',
        help='extension of caption file, default is `.txt`'
    )
    joy_args.add_argument(
        '--joy_read_wd_caption',
        action='store_true',
        help='joy will read wd caption for inference.\nOnly effect when `caption_method` is `joy`'
    )
    joy_args.add_argument(
        '--joy_caption_without_wd',
        action='store_true',
        help='joy will not read wd caption for inference.\nOnly effect when `caption_method` is `both`'
    )
    joy_args.add_argument(
        '--joy_user_prompt',
        type=str,
        default=DEFAULT_USER_PROMPT_WITHOUT_WD,
        help='user prompt for caption.'
    )
    joy_args.add_argument(
        '--joy_temperature',
        type=float,
        default=0.5,
        help='temperature for joy LLM model, default is `0.5`.'
    )
    joy_args.add_argument(
        '--joy_max_tokens',
        type=int,
        default=300,
        help='max tokens for joy LLM model output, default is `300`.'
    )

    caption_args = args.add_argument_group("Caption")
    caption_args.add_argument(
        '--caption_method',
        type=str,
        default='both',
        choices=['both', 'wd', 'joy'],
        help='method for caption[`both`, `wd`, `joy`],select wd or joy models, or both of them to caption, '
             'default is `both`.',
    )
    caption_args.add_argument(
        '--run_method',
        type=str,
        default='sync',
        choices=['sync', 'queue'],
        help='''running method for wd+joy caption[`sync`, `queue`], need `caption_method` set to `both`.
             if sync, image will caption with wd models,
             then caption with joy models while wd captions in joy user prompt.
             if queue, all images will caption with wd models first,
             then caption all of them with joy models while wd captions in joy user prompt.
             default is `sync`.'''
    )
    caption_args.add_argument(
        '--image_size',
        type=int,
        default=1024,
        help='resize image to suitable, default is `1024`.'
    )
    caption_args.add_argument(
        '--not_overwrite',
        action='store_true',
        help='not overwrite caption file if exists.'
    )
    caption_args.add_argument(
        '--custom_caption_save_path',
        type=str,
        default=None,
        help='custom caption file save path.'
    )

    return args

if __name__ == "__main__":
    get_args = setup_args()
    get_args = get_args.parse_args()
    main(get_args)
