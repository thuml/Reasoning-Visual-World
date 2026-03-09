
from openai import OpenAI
import time
import json
from io import BytesIO
import PIL.Image


def pil_image_to_base64(image, format: str = "PNG") -> str:
    import base64
    # 创建内存缓冲区
    buffer = BytesIO()

    # 保存图片到缓冲区（需指定格式，注意：JPEG 不支持透明通道，透明图建议用 PNG）
    image.save(buffer, format=format)

    # 将缓冲区指针移到开头，以便读取完整数据
    buffer.seek(0)

    # 读取二进制数据并编码为 base64（转为字符串返回）
    return base64.b64encode(buffer.read()).decode("utf-8")


def get_client(ak):
    client = OpenAI(
        api_key=ak,
        base_url="EMPTY",  # TODO: add your API server here
        timeout=3600
    )
    return client


def get_response(client, message_content):
    response = client.chat.completions.create(
        model="MODEL_NAME",   # TODO: add your model name here
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ],
        max_tokens=32768,
    )
    full_resp = json.loads(response.model_dump_json())["choices"][0]["message"]["content"]
    return full_resp


def get_response_until(get_response, client, message_content):
    while True:
        try:
            response = get_response(client, message_content)
            return response
        except Exception as e:
            print(e)
            time.sleep(10)


def get_message(raw_message_content, images):
    message_content = []
    for content in raw_message_content:
        if content in images:
            message_content.append(
                {"type": "input_image", "image_url": f"data:image/png;base64,{pil_image_to_base64(content)}"})
        else:
            message_content.append({"type": "input_text", "text": content})
    return message_content
