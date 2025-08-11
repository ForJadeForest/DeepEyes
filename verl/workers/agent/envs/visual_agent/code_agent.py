import base64
import json
import os
import random
import re
import socket
import uuid
from io import BytesIO
from math import ceil, floor
from time import sleep
from typing import Union

import numpy as np
import requests
from PIL import Image

from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents


def get_local_ip():
    """
    获取本地IP地址
    :return: 本地IP地址字符串，或者在无法获取时返回 None
    """
    s = None
    try:
        # 创建一个UDP套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 尝试连接到一个公共的IP地址（不需要是可达的）
        # 这里的8.8.8.8是Google的DNS服务器，端口号可以任意选择
        s.connect(('8.8.8.8', 80))
        # getsockname()返回套接字自己的地址
        ip = s.getsockname()[0]
    except Exception as e:
        print(f"无法获取IP地址: {e}")
        ip = None
    finally:
        if s:
            s.close()
    return ip

default_sandbox_url="http://29.208.50.62:16384"
local_ip = get_local_ip()
if local_ip is None:
    SANDBOX_URL = default_sandbox_url
else:
    SANDBOX_URL = f"http://{local_ip}:8080"

print(SANDBOX_URL)

def encode_image_path_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """Encode image to base64 string"""
    if isinstance(image, str):
        return encode_image_path_base64(image)
    elif isinstance(image, Image.Image):
        return encode_pil_image_to_base64(image)
    else:
        raise ValueError("Image must be a file path or PIL Image instance")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # 验证图像尺寸
    width, height = image.size
    
    # 检查宽和高不能小于28
    if width < 28 or height < 28:
        raise ValueError(f"Image size too small: {width}x{height}. Minimum size is 28x28.")
    
    # 检查宽高比不能>200
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 200:
        raise ValueError(f"Image aspect ratio too extreme: {aspect_ratio:.2f}. Maximum allowed is 200.")
    
    return image 


def run_jupyter_code(cell_list, sandbox_url, upload_file_dict=None, max_retries=2, retry_delay=1):
    """
    Run Jupyter code with retry mechanism.
    
    Args:
        cell_list: List of code cells to execute
        sandbox_url: URL of the sandbox service
        upload_file_dict: Optional dictionary of files to upload
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1)
    
    Returns:
        output_cells: List of output cells from Jupyter execution
    
    Raises:
        ValueError: If no output cells returned after all retries
        requests.RequestException: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            response = requests.post(
                f"{sandbox_url}/run_jupyter",
                json={
                    "cells": cell_list,
                    "kernel": "python3",
                    "files": upload_file_dict,
                    "total_timeout": 45,
                },
                timeout=50,  # Add request timeout
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            output_cells = response.json().get("cells", [])
            if not output_cells:
                raise ValueError(
                    f"No output cells returned from Jupyter execution. Cell List: {cell_list}"
                )
            
            return output_cells
            
        except (requests.RequestException, ValueError) as e:
            last_exception = e
            if attempt < max_retries:
                print(f" [DEBUG] Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
                # Exponential backoff: increase delay for next retry
                retry_delay *= 2
            else:
                print(f" [DEBUG] All {max_retries + 1} attempts failed. Last error: {e}")
    
    # If we get here, all retries failed
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError("All retry attempts failed with unknown error")


def parse_cell_output(cell_output: dict) -> dict:
    """Parse the output of a Jupyter cell."""
    if not cell_output:
        return {
            "text_output": "",
            "image_output": [],
            "has_error": False,
        }
    stdout = cell_output.get("stdout", "")

    errors = cell_output.get("error", "")
    error_message = ""
    for e in errors:
        traceback = e.get("traceback", "")
        e_name = e.get("ename", "")
        e_value = e.get("evalue", "")
        error_message = f"[CODE RUN ERROR]: {e_name} - {e_value}\nTraceback: {traceback}"

    # show display output
    display_output = cell_output.get("display", [])
    display_text = ""
    display_image = []
    for cell_output_item in display_output:
        for key, value in cell_output_item.items():
            if key == "text/plain":
                display_text += value
            elif key == "image/png":
                display_image.append(f"data:image/png;base64,{value}")
            elif key == "image/jpeg":
                display_image.append(f"data:image/jpeg;base64,{value}")
            else:
                print(f"Unknown key: {key}")
    text_output = ""
    if stdout:
        text_output += f"stdout: {stdout}\n"
    if display_text:
        text_output += f"display text: {display_text}\n"
    if error_message:
        text_output += f"error: {error_message}\n"

    image_output = display_image
    if "<image>" in text_output:
        print("[ERROR-LEVEL0] Found <image> in text output, indicating image generation.")
        print(cell_output)
        text_output = text_output.replace("<image>", "")
        print("[ERROR-LEVEL0] Replaced <image> in text output with empty string.")
    return {
        "text_output": text_output.strip(),
        "image_output": image_output,
        "has_error": bool(error_message),
    }


def cell_output_to_str(cell_output: dict) -> dict:
    text_output = cell_output["text_output"]
    image_output = cell_output["image_output"]
    if image_output:
        try:
            # 尝试转换所有图像，验证其尺寸
            valid_images = []
            for img in image_output:
                try:
                    valid_images.append(base64_to_image(img.split(",", 1)[-1]))
                except ValueError as e:
                    # 如果图像验证失败，记录错误但继续处理其他图像
                    print(f" [DEBUG] Image validation failed: {e}")
                    continue
            
            if valid_images:
                prompt = f"<interpreter>{text_output}\n{'<|vision_start|><|image_pad|><|vision_end|>' * len(valid_images)}\n</interpreter>"
                return {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": valid_images
                    },
                    "has_error": cell_output.get("has_error", False),
                }
            else:
                # 如果所有图像都无效，返回纯文本结果
                prompt = f"<interpreter>{text_output}\n[Note: Generated images were invalid because the image size is too small or the aspect ratio [max(height, width) / min(height, width) >= 200] is incorrect]\n</interpreter>"
                return {"prompt": prompt, "multi_modal_data": {"image": []}, "code_error": cell_output.get("has_error", False)}
        except Exception as e:
            # 如果出现其他错误，返回纯文本结果
            print(f" [DEBUG] Error processing images: {e}")
            prompt = f"<interpreter>{text_output}\n[Error: Failed to process generated images. Error: {e}]\n</interpreter>"
            return {"prompt": prompt, "multi_modal_data": {"image": []}, "code_error": cell_output.get("has_error", False)}
    else:
        prompt = f"<interpreter>{text_output}\n</interpreter>"
        return {"prompt": prompt, "multi_modal_data": {"image": []}, "code_error": cell_output.get("has_error", False)}


class CodeAgentEnv(ToolBase):
    name = "CodeAgentEnv"
    action_start = "<tool_call>"
    action_end = "</tool_call>"
    answer_start = "<answer>"
    answer_end = "</answer>"
    sandbox_url = SANDBOX_URL

    # <tool_call>\n{"name": "zoom_in", "arguments": {"object": "woman\'s jacket"}}\n</tool_call>

    chat_template = """<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""

    def __init__(self, _name, _desc, _params, **kwargs):
        self.chatml_history = []
        self.multi_modal_data = None
        self.code_list = []
        self.upload_file_dict = {}
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        answers = extract_tool_call_contents(
            self.answer_start, self.answer_end, action_string
        )
        if answers:
            # print(f' [DEBUG] found answer in {action_string=}')
            return "", 0.0, True, {}

        action_list = extract_tool_call_contents(
            self.action_start, self.action_end, action_string
        )
        if not action_list:
            # print(f' [DEBUG] no action_list in {action_string=}')
            return "", 0.0, True, {}

        action = [action.strip() for action in action_list]
        if len(action_list) != 1:
            print(f" [DEBUG] found {len(action_list)} actions in {action_string=}, but expected 1")
            user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: Only one tool call action is allowed.\n</tool_response>")
            return user_msg, -1.0, False, {}
        action = action[0]
        try:
            try:
                action_json = json.loads(action)
            except json.JSONDecodeError:
                print(f" [DEBUG] Error decoding JSON: {action}")
                user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: Function json string are invalid. Can not be loads by json.loads().\n</tool_response>")
                return user_msg, -1.0, False, {}
            if "arguments" not in action_json:
                print(f" [DEBUG] 'arguments' parameter not in json string {action_json}")
                user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: 'arguments' parameter not in json string\n</tool_response>")
                return user_msg, -1.0, False, {}
            if "code" not in action_json["arguments"]:
                print(f" [DEBUG] 'code' parameter not in arguments {action_json['arguments']}")
                user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: 'code' parameter not in arguments.\n</tool_response>")
                return user_msg, -1.0, False, {}
            
            code = action_json["arguments"]["code"]
            if isinstance(code, str):
                code = code.strip()
            else:
                print(f" [DEBUG] 'code' is not a string: {type(code)}")
                user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: 'code' parameter need to be a string.\n</tool_response>")
                return user_msg, -1.0, False, {}
            if not code:
                print(" [DEBUG] 'code' is empty")
                user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: not find code in arguments</tool_response>")
                return user_msg, -1.0, False, {}

        except Exception as e:
            print(f" [ERROR] Error parsing code: {e} with action: {action}")
            user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: Function ARGUMENTS ARE INVALID. Error: " + str(e) + "\n</tool_response>")
            return user_msg, -1.0, False, {}

        self.code_list.append(code)

        # TODO: modify here and process the final output
        try:
            cell_out = run_jupyter_code(
                self.code_list,
                sandbox_url=self.sandbox_url,
                upload_file_dict=self.upload_file_dict,
            )
        except Exception as err:
            print(f" [ERROR] run_jupyter_code error: {err}, {self.code_list=}")
            user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: Call jupyter service error\n</tool_response>")
            return user_msg, -1.0, False, {}
        try:
            parsed_output = parse_cell_output(cell_out[-1])
            code_output = cell_output_to_str(parsed_output)
        except Exception as err:
            print(f" [ERROR] Error parsing Jupyter cell output: {err}")
            user_msg = self.chat_template.format("<tool_response>\n[TOOLERROR]: Error parsing Jupyter cell output: " + str(err) + "\n</tool_response>")
            return user_msg, -1.0, False, {}

        all_user_msg = self.chat_template.format(f'<tool_response>\n{code_output["prompt"]})\n</tool_response>')
        obs_dict = {
            "prompt": all_user_msg,
            "multi_modal_data": code_output["multi_modal_data"],
            "has_error": code_output.get("has_error", False),
        }
        if code_output.get("has_error", False):
            return obs_dict, -1.0, False, {}
        return obs_dict, 0.0, False, {}

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        image_id = kwargs.get("image_id", None)
        if image_id is None:
            raise ValueError("image_id must be provided in reset kwargs")
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert (
            "image" in self.multi_modal_data.keys()
        ), f"[ERROR] {origin_multi_modal_data=}"
        assert (
            len(self.multi_modal_data["image"]) > 0
        ), f'[ERROR] {self.multi_modal_data["image"]=}'
        self.height = self.multi_modal_data["image"][0].height
        self.width = self.multi_modal_data["image"][0].width
        self.upload_file_dict = {
            f"./{image_id}.jpg": encode_image_base64(self.multi_modal_data["image"][0])
        }
