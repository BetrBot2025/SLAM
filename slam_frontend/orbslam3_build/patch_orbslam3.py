"""WP2: minimal source patch adding a public keyframe signal to ORB-SLAM3.

Vanilla ORB-SLAM3's System class has no way to tell a caller whether the
frame just processed became a keyframe -- that decision (NeedNewKeyFrame()/
CreateNewKeyFrame()) is entirely internal to Tracking. Algorithm 1 of
DSP-SLAM++ needs exactly that signal to gate when to run 2D/3D detection.
This is the same kind of source-level extension the original DSP-SLAM made
to ORB-SLAM2/3 for the same reason.

Run against a freshly-cloned ORB_SLAM3 checkout, before building:
    python3 patch_orbslam3.py /opt/ORB_SLAM3
"""
import sys


def replace_once(path: str, old: str, new: str):
    with open(path, "r") as f:
        content = f.read()
    if old not in content:
        raise SystemExit(f"patch anchor not found in {path}:\n{old!r}")
    if content.count(old) != 1:
        raise SystemExit(f"patch anchor not unique in {path} ({content.count(old)} occurrences):\n{old!r}")
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"patched {path}")


def main(root: str):
    # -- Tracking.h: declare the flag + a public accessor -------------------
    replace_once(
        f"{root}/include/Tracking.h",
        "    KeyFrame* GetLastKeyFrame()\n"
        "    {\n"
        "        return mpLastKeyFrame;\n"
        "    }\n",
        "    KeyFrame* GetLastKeyFrame()\n"
        "    {\n"
        "        return mpLastKeyFrame;\n"
        "    }\n"
        "\n"
        "    // WP2 patch: exposes ORB-SLAM3's internal keyframe decision.\n"
        "    bool WasLastFrameAKeyFrame() const { return mbLastFrameWasKeyFrame; }\n",
    )
    replace_once(
        f"{root}/include/Tracking.h",
        "    int mTrackedFr;\n"
        "    bool mbStep;\n",
        "    int mTrackedFr;\n"
        "    bool mbStep;\n"
        "    bool mbLastFrameWasKeyFrame = false;   // WP2 patch\n",
    )

    # -- Tracking.cc: reset per-frame, set at the real creation call site ---
    replace_once(
        f"{root}/src/Tracking.cc",
        "void Tracking::Track()\n"
        "{\n"
        "\n"
        "    if (bStepByStep)\n",
        "void Tracking::Track()\n"
        "{\n"
        "    mbLastFrameWasKeyFrame = false;   // WP2 patch\n"
        "\n"
        "    if (bStepByStep)\n",
    )
    replace_once(
        f"{root}/src/Tracking.cc",
        "                CreateNewKeyFrame();\n",
        "                CreateNewKeyFrame();\n"
        "                mbLastFrameWasKeyFrame = true;   // WP2 patch\n",
    )

    # -- System.h/.cc: forward the accessor through System ------------------
    replace_once(
        f"{root}/include/System.h",
        "    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();\n",
        "    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();\n"
        "    bool WasLastFrameAKeyFrame();   // WP2 patch\n",
    )
    replace_once(
        f"{root}/src/System.cc",
        "vector<MapPoint*> System::GetTrackedMapPoints()\n",
        "bool System::WasLastFrameAKeyFrame()   // WP2 patch\n"
        "{\n"
        "    return mpTracker->WasLastFrameAKeyFrame();\n"
        "}\n"
        "\n"
        "vector<MapPoint*> System::GetTrackedMapPoints()\n",
    )

    print("All patches applied.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/opt/ORB_SLAM3")
