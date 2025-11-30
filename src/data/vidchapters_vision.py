from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

from src.data.prompt import Prompt
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class VidChaptersVision(Dataset):
    def __init__(self, prompter: Prompt, **kwargs):
        self.prompter = prompter
        self.chapters = self.prompter.chapters

        video_ids = []
        for vid_id in self.chapters.video_ids:
            if vid_id in self.prompter:
                video_ids.append(vid_id)

        assert len(video_ids) > 0, (
            f"No {self.prompter.chapters.subset} videos found for {self.prompter.__class__.__name__} at {self.prompter.chapters.embs_dir}"
        )
        vid_found = len(video_ids) / len(self.chapters.video_ids)
        prompter_name = self.prompter.__class__.__name__
        log.info(f"Found {vid_found:.2%} videos for {prompter_name}")
        self.video_ids = video_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        prompt = self.prompter.get_prompt_test(vid_id)
        vid_duration = self.chapters.get_duration(vid_id, hms=True)
        transcript = self.prompter.get_transcript_test(vid_id)

        return {
            "vid_id": vid_id,
            "prompt": prompt,
            "vid_duration": vid_duration,
            "transcript": transcript,
        }

    # Used for training
    def process(self, tokenize_dialog, tokenizer):
        # function only used for training
        dialogs = []
        vid_ids = []
        for vid_id in self.chapters.video_ids:
            if vid_id in self.prompter:
                dialogs.append(self.prompter.get_dialog(vid_id))
                vid_ids.append(vid_id)

        dataset = HFDataset.from_dict({"dialog": dialogs, "vid_id": vid_ids})

        dataset = dataset.map(
            lambda x: tokenize_dialog(x["dialog"], tokenizer),
            remove_columns=["dialog"],
        )
        return dataset

    # Used for inference
    def test_dataloader(self, batch_size: int = 1, num_workers: int = 4):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    from src.data.utils_frames import ChaptersFrames, PromptFrames

    embs_dir = "dataset/embs/google/siglip-so400m-patch14-384/HwwwH_MiniCPM-V-2/asr_s10k-2_train_preds+no-asr-10s"
    chapters = ChaptersFrames(
        embs_dir=embs_dir, subset="s1k_train", vision_feature_select_strategy="full"
    )
    prompter = PromptFrames(chapters=chapters)
    data = VidChaptersVision(prompter)

    for batch in data.test_dataloader():
        print(batch["vid_id"][0])
        print(batch["vid_duration"][0])
        print(batch["prompt"][0], end="")
        print(batch["transcript"][0])
        break
