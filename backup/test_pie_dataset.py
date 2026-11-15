

from pie_data_Hesam import PIE
import torch

if __name__ == '__main__':

    imdb = PIE(data_path=r'D:\PIE codes\PIE_dataset')

    # ── Test split (719 image-usable tracks) ──────────────────────────────
    test_ds = PIEIntentionDataset(
        imdb, 'test',
        observe_length=15,
        backbone='vgg16',
        seq_overlap_rate=0.5,
        normalize_bbox=True,
    )

    # 1. Sliding window must expand tracks into many more windows.
    print(f"\ntracks={len(test_ds.tracks)}  windows={len(test_ds)}")
    assert len(test_ds) > len(test_ds.tracks), \
        "FAIL: windows should be >> tracks. Sliding window not working."

    # 2. Single-sample shape check.
    s = test_ds[0]
    assert s['features'].shape == (15, 4096),  f"features shape wrong: {s['features'].shape}"
    assert s['bbox'].shape     == (15, 4),     f"bbox shape wrong: {s['bbox'].shape}"
    assert float(s['bbox'].max()) <= 1.0,      f"bbox not normalized: max={s['bbox'].max()}"
    assert s['label'].dtype    == torch.int64, f"label dtype wrong: {s['label'].dtype}"
    print(f"features: {s['features'].shape} {s['features'].dtype}")
    print(f"bbox:     {s['bbox'].shape}  max={float(s['bbox'].max()):.4f}")
    print(f"label:    {s['label']}  dtype={s['label'].dtype}")
    print(f"int_prob: {s['intention_prob']}")
    print(f"ped_id:   {s['ped_id']}")

    # 3. DataLoader batch check.
    from torch.utils.data import DataLoader
    dl    = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    assert batch['features'].shape == (32, 15, 4096)
    assert batch['bbox'].shape     == (32, 15, 4)
    assert batch['label'].shape    == (32,)
    assert len(batch['ped_id'])    == 32
    print(f"\nbatch features: {batch['features'].shape}")
    print(f"batch bbox:     {batch['bbox'].shape}")
    print(f"batch label:    {batch['label'].shape}")
    print(f"batch ped_id:   {len(batch['ped_id'])} strings")

    # 4. Second instantiation must be cache hit (no GPU work).
    print("\n── Second instantiation (expect: all cached) ──")
    test_ds2 = PIEIntentionDataset(imdb, 'test', observe_length=15)

    # 5. Train and val splits.
    train_ds = PIEIntentionDataset(imdb, 'train', observe_length=15)
    val_ds   = PIEIntentionDataset(imdb, 'val',   observe_length=15)

    print("\n── Phase 2 END-TO-END: ALL CHECKS PASSED ──")
