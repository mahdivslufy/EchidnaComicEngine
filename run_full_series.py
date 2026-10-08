import os
import sys
import re
import time
import zipfile
import shutil
from typing import Optional

sys.path.append(r"C:\Users\ITcenter\EchidnaComicEngine")
from main import process_single_image

cbz_path = r"C:\Users\ITcenter\Downloads\Rick and Morty - vs. Cthulhu (2023) (Digital) (danke-Emp.cbz"
work_dir = r"C:\Users\ITcenter\ComicProjects\EchidnaEngine_Full_Series"
raw_dir = os.path.join(work_dir, "raw")
trans_dir = os.path.join(work_dir, "translated")
dl_dir = r"C:\Users\ITcenter\Downloads"

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(trans_dir, exist_ok=True)

def translate_issue(issue_prefix: str, start_idx: int = 0, end_idx: Optional[int] = None):
    with zipfile.ZipFile(cbz_path, "r") as z:
        all_files = sorted([f for f in z.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
        issue_files = [f for f in all_files if issue_prefix in f]
        total_issue = len(issue_files)
        target_files = issue_files[start_idx:end_idx] if end_idx is not None else issue_files[start_idx:]
        
        print(f"\n========================================================")
        print(f"  ECHIDNA ENGINE PRO: {issue_prefix.upper()} ({len(target_files)}/{total_issue} pages)")
        print(f"========================================================")

        for idx, file_name in enumerate(target_files, start_idx + 1):
            base_name = os.path.basename(file_name)
            m = re.search(r"p\d+", base_name)
            pg_str = m.group(0) if m else f"p{idx:03d}"
            out_name = f"Rick_and_Morty_{issue_prefix}_{pg_str}_PRO_FA.jpg"

            raw_target = os.path.join(raw_dir, base_name)
            out_target = os.path.join(trans_dir, out_name)

            if not os.path.exists(raw_target):
                with open(raw_target, "wb") as f_out:
                    f_out.write(z.read(file_name))

            if os.path.exists(out_target):
                print(f"[{idx}/{total_issue}] Already translated: {out_name}")
                continue

            print(f"[{idx}/{total_issue}] Translating: {base_name[:55]}... -> {out_name}")
            t0 = time.time()
            try:
                process_single_image(raw_target, out_target)
                print(f"[✓] Finished in {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"[!] Error: {e}. Retrying with fallback...")
                time.sleep(2)
                try:
                    process_single_image(raw_target, out_target)
                    print(f"[✓] Finished on retry in {time.time()-t0:.2f}s")
                except Exception as e2:
                    print(f"[X] Page failed: {e2}. Keeping raw.")
                    shutil.copyfile(raw_target, out_target)

        # Repack issue CBZ
        issue_cbz = os.path.join(dl_dir, f"Rick_and_Morty_vs_Cthulhu_{issue_prefix}_Persian_PRO.cbz")
        print(f"\n[*] Packaging {issue_prefix.upper()} into {issue_cbz}...")
        with zipfile.ZipFile(issue_cbz, "w", zipfile.ZIP_DEFLATED) as z_out:
            for f in sorted(os.listdir(trans_dir)):
                if issue_prefix in f and f.endswith(".jpg"):
                    z_out.write(os.path.join(trans_dir, f), arcname=f)
        print(f"[✓] {issue_prefix.upper()} CBZ ready: {issue_cbz} ({os.path.getsize(issue_cbz)/(1024*1024):.1f} MB)")


def package_complete_omnibus():
    omnibus_cbz = os.path.join(dl_dir, "Rick_and_Morty_vs_Cthulhu_COMPLETE_Persian_PRO.cbz")
    print(f"\n[*] Packaging ALL issues into Complete Omnibus CBZ: {omnibus_cbz}...")
    files = sorted([f for f in os.listdir(trans_dir) if f.endswith(".jpg")])
    with zipfile.ZipFile(omnibus_cbz, "w", zipfile.ZIP_DEFLATED) as z_out:
        for idx, f in enumerate(files):
            arcname = f"{idx:03d}_{f}"
            z_out.write(os.path.join(trans_dir, f), arcname=arcname)
    size_mb = os.path.getsize(omnibus_cbz) / (1024 * 1024)
    print(f"[✓] COMPLETE OMNIBUS READY! Total Pages: {len(files)} | Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        for iss in ["c001", "c002", "c003", "c004"]:
            translate_issue(iss)
        package_complete_omnibus()
    elif target in ["c001", "c002", "c003", "c004"]:
        s = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        e = int(sys.argv[3]) if len(sys.argv) > 3 else None
        translate_issue(target, s, e)
    elif target == "pack":
        package_complete_omnibus()
