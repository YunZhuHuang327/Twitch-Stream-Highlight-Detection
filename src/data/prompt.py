from src.data.chapters import Chapters, sec_to_hms


class Prompt:
    def __init__(
        self,
        chapters: Chapters,
    ):
        self.chapters = chapters

    def __contains__(self, vid_id):
        raise NotImplementedError(
            "Subclasses must implement the '__contains__' method."
        )

    def get_duration_prompt(self, vid_id: str) -> str:
        duration = self.chapters.get_duration(vid_id, hms=True)
        return f"Given the complete transcript of a video of duration {duration}, "

    def get_task_prompt(self) -> str:
        raise NotImplementedError(
            "Subclasses must implement the 'get_task_prompt' method."
        )

    def get_format_instruction(self):
        return "Identify the approximate start time of each chapter in the format 'hh:mm:ss - Title'. "

    def get_new_line_instruction(self):
        return "Ensure each chapter entry is on a new line. "

    def get_focus_instruction(self):
        return "Focus on significant topic changes that would merit a new chapter in a video, "

    def get_no_summaries_instruction(self):
        return "but do not provide summaries of the chapters.\n"

    def get_transcript_introduction(self):
        return "Here is the transcript to analyze:\n"

    def get_transcript(self, vid_id: str) -> str:
        # By default, the transcript is the same for train and test
        raise NotImplementedError(
            "Subclasses must implement the 'get_transcript' method."
        )

    def get_transcript_train(self, vid_id: str) -> str:
        return self.get_transcript(vid_id)

    def get_transcript_test(self, vid_id: str) -> str:
        return self.get_transcript(vid_id)

    def get_base_prompt(self, vid_id: str) -> str:
        prompt_parts = [
            self.get_duration_prompt(vid_id),
            self.get_task_prompt(),
            self.get_format_instruction(),
            self.get_new_line_instruction(),
            self.get_focus_instruction(),
            self.get_no_summaries_instruction(),
            self.get_transcript_introduction(),
        ]
        return "".join(prompt_parts)

    def get_prompt_train(self, vid_id: str) -> str:
        return self.get_base_prompt(vid_id)

    def get_prompt_test(self, vid_id: str) -> str:
        return self.get_base_prompt(vid_id)

    def get_output(self, vid_id: str) -> str:
        vid_chapters = self.chapters.get_chapters(vid_id)
        answers = []
        for chp_time, chp_title in vid_chapters.items():
            chp_time = sec_to_hms(chp_time)
            answers.append(f"{chp_time} - {chp_title}")

        return "\n".join(answers)

    def get_dialog(self, vid_id: str) -> str:
        prompt = self.get_prompt_train(vid_id)
        transcript = self.get_transcript_train(vid_id)
        output = self.get_output(vid_id)
        dialog = [
            {
                "role": "user",
                "content": prompt + transcript,
            },
            {
                "role": "assistant",
                "content": output,
            },
        ]
        return dialog
