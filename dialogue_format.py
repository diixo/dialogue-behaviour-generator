"""Runtime text formatting. Dataset files contain no format-specific markers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DialogueFormat:
    """Templates use {role} and {content}; the dataset ensures EOS after any suffix."""

    message_template: str = '{role}: {content}\n'
    assistant_prefix: str = 'assistant: '
    completion_suffix: str | None = None

    def render(self, messages, completion):
        prompt = ''.join(self.message_template.format(**message) for message in messages)
        return {
            'prompt': prompt + self.assistant_prefix,
            'completion': completion + (self.completion_suffix or ''),
        }


FORMATS = {
    'plain': DialogueFormat(),
    'chatml': DialogueFormat(
        message_template='<|im_start|>{role}\n{content}<|im_end|>\n',
        assistant_prefix='<|im_start|>assistant\n',
        completion_suffix='<|im_end|>',
    ),
}


def get_format(style='plain'):
    if isinstance(style, DialogueFormat):
        return style
    try:
        return FORMATS[style]
    except KeyError:
        raise ValueError(f'Unknown dialogue style: {style!r}; expected one of {tuple(FORMATS)}') from None


def format_example(example, style='plain'):
    """Render a structured SFT record without modifying the source messages."""
    return get_format(style).render(example['messages'], example['completion'])
