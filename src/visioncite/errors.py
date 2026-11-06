"""项目统一异常。

企业代码里"错误也是接口的一部分"：调用方应当能区分
"用户传错文件"和"服务器缺少模型"，从而给出不同的提示或降级策略。
所以这里用一个小而清晰的异常树，而不是到处 ``raise Exception``。
"""

from __future__ import annotations


class VisionCiteError(Exception):
    """所有本项目异常的基类。

    Attributes:
        message: 面向开发者的说明。
        user_message: 面向终端用户的中文提示，可以被 API 层直接展示。
    """

    default_user_message = "处理失败，请稍后重试。"

    def __init__(self, message: str, user_message: str | None = None) -> None:
        """初始化异常。

        Args:
            message: 开发者看的详细信息。
            user_message: 用户看的一句话；为空时使用类默认值。
        """
        super().__init__(message)
        self.message = message
        self.user_message = user_message or self.default_user_message


class UnsupportedFileError(VisionCiteError):
    """文件类型不支持（例如上传了 .docx 或压缩包）。"""

    default_user_message = "暂不支持这种文件类型，请上传 PDF 或图片。"


class IngestError(VisionCiteError):
    """解析失败：文件损坏、加密 PDF、渲染失败等。"""

    default_user_message = "文件解析失败，请检查文件是否损坏或加了密码。"


class OcrUnavailableError(VisionCiteError):
    """OCR 后端不可用（未安装 rapidocr）。属于可降级错误。"""

    default_user_message = "当前环境未启用 OCR，只能读取 PDF 文本层。"


class VisualUnavailableError(VisionCiteError):
    """视觉编码器不可用（未安装 transformers/权重缺失）。属于可降级错误。"""

    default_user_message = "当前环境未启用视觉检索，仅使用文本检索。"


class AnswerBackendError(VisionCiteError):
    """外部 VLM 调用失败（超时、限流、返回格式非法）。"""

    default_user_message = "生成答案失败，请稍后再试。"


class VerificationError(VisionCiteError):
    """答案校验不通过（引用不存在、数字对不上、支撑不足）。

    注意：这不是"程序错误"，而是安全网在工作 —— 上层会把它转成"弃答"。
    """

    default_user_message = "没有找到足够可靠的证据，因此不作回答。"
