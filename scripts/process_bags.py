#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import h5py
import bisect
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm.auto import tqdm

# -------------------------
# Configuration
# -------------------------
SYNC_TOLERANCE_NS = 50 * 1e6  # 50 milliseconds tolerance
MIN_MARKER_DISTANCE_M = 0.02  # 2cm minimum distance between consecutive markers
SELF_INTERSECTION_THRESHOLD_M = 0.015  # 1.5cm - if non-adjacent segments are closer than this, wire is self-intersecting
# PIVOT_TOPIC = '/digit/D21122/image_raw_throttled' # The topic used as the "center" of the timeframe

TOPIC_MAP = {
    '/end_effector_pose_throttled': 'ee',
    '/natnet/fixed_ee_pose_throttled': 'fixed',
    '/joint_states_throttled': 'joint_states',
    '/digit/D21118/image_raw_throttled': 'digit_1',
    '/digit/D21122/image_raw_throttled': 'digit_2',
    '/digit/D21123/image_raw_throttled': 'digit_3',
    '/digit/D21124/image_raw_throttled': 'digit_4',
    '/natnet/unlabeled_marker_data_throttled': 'markers',
}
TOPIC_MAP_FULL = {
    '/full_data/end_effector_pose': 'ee',
    '/full_data/natnet/fixed_ee_pose': 'fixed',
    '/full_data/joint_states': 'joint_states',
    '/full_data/digit/D21118/image_raw': 'digit_1',
    '/full_data/digit/D21122/image_raw': 'digit_2',
    '/full_data/digit/D21123/image_raw': 'digit_3',
    '/full_data/digit/D21124/image_raw': 'digit_4',
    '/full_data/natnet/unlabeled_marker_data': 'markers',
}
# Older bags (25_12, 28_12, 29_12) use different topic names for poses
TOPIC_MAP_FULL_OLD = {
    '/full_data/natnet_robot_ee_pose': 'ee',
    '/full_data/natnet_fixed_pose': 'fixed',
    '/full_data/joint_states': 'joint_states',
    '/full_data/digit/D21118/image_raw': 'digit_1',
    '/full_data/digit/D21122/image_raw': 'digit_2',
    '/full_data/digit/D21123/image_raw': 'digit_3',
    '/full_data/digit/D21124/image_raw': 'digit_4',
    '/full_data/natnet/unlabeled_marker_data': 'markers',
}

# -------------------------
# Helper: Find closest timestamp
# -------------------------
def find_closest_msg(target_ts: int, buffer_list: List[Tuple[int, Any]], tolerance: int, used_indices: Optional[set] = None) -> Optional[Tuple[int, Any]]:
    """
    Uses binary search to find the message in buffer_list closest to target_ts.
    buffer_list must be sorted by timestamp.
    
    Args:
        target_ts: Target timestamp to match
        buffer_list: List of (timestamp, message) tuples, sorted by timestamp
        tolerance: Maximum allowed time difference in nanoseconds
        used_indices: Set of indices that have already been matched (to ensure each msg is used once)
    
    Returns:
        Tuple of (index, message) if found, None otherwise
    """
    if not buffer_list:
        return None

    if used_indices is None:
        used_indices = set()

    # Extract just timestamps for bisecting
    timestamps = [x[0] for x in buffer_list]
    
    # Find insertion point
    idx = bisect.bisect_left(timestamps, target_ts)
    
    candidates = []
    
    # Check left neighbor (if not already used)
    if idx > 0 and (idx - 1) not in used_indices:
        candidates.append((idx - 1, abs(timestamps[idx - 1] - target_ts)))
    
    # Check current/right neighbor (if not already used)
    if idx < len(timestamps) and idx not in used_indices:
        candidates.append((idx, abs(timestamps[idx] - target_ts)))

    if not candidates:
        return None

    # Pick the closest one
    best_idx, best_diff = min(candidates, key=lambda x: x[1])

    if best_diff <= tolerance:
        return (best_idx, buffer_list[best_idx][1])  # Return (index, msg)
    
    return None


def segment_min_distance(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Compute the minimum distance between two 3D line segments.
    Segment 1: p1 to p2
    Segment 2: p3 to p4
    """
    d1 = p2 - p1  # Direction of segment 1
    d2 = p4 - p3  # Direction of segment 2
    r = p1 - p3
    
    a = np.dot(d1, d1)  # |d1|^2
    e = np.dot(d2, d2)  # |d2|^2
    f = np.dot(d2, r)
    
    # Check for degenerate segments (points)
    if a < 1e-10 and e < 1e-10:
        return np.linalg.norm(r)
    if a < 1e-10:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    elif e < 1e-10:
        t = 0.0
        s = np.clip(-np.dot(d1, r) / a, 0.0, 1.0)
    else:
        b = np.dot(d1, d2)
        c = np.dot(d1, r)
        denom = a * e - b * b
        
        if abs(denom) > 1e-10:
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
        else:
            s = 0.0
        
        t = (b * s + f) / e
        
        if t < 0.0:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = np.clip((b - c) / a, 0.0, 1.0)
    
    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    return np.linalg.norm(closest1 - closest2)


def check_self_intersection(pts: np.ndarray, threshold: float) -> bool:
    """
    Check if a polyline (defined by pts) has self-intersections.
    Two segments are considered intersecting if they are non-adjacent
    and their minimum distance is less than threshold.
    
    Args:
        pts: Nx3 array of 3D points defining the polyline
        threshold: Distance threshold for intersection detection
    
    Returns:
        True if self-intersection detected, False otherwise
    """
    n_segments = len(pts) - 1
    
    for i in range(n_segments):
        # Only check segments that are at least 2 apart (skip adjacent segments)
        for j in range(i + 2, n_segments):
            # Skip if segments share a vertex (adjacent)
            if j == i + 1:
                continue
            
            dist = segment_min_distance(pts[i], pts[i+1], pts[j], pts[j+1])
            if dist < threshold:
                return True
    
    return False

# -------------------------
# Reader Class
# -------------------------
class ApproximateBagReader:
    def __init__(self):
        # Will be populated dynamically based on detected topics
        self.buffers: Dict[str, List[Tuple[int, Any]]] = {}
        self.topic_map = None

    def _detect_topic_map(self, reader: AnyReader) -> Dict[str, str]:
        """Detect which topic map to use based on topics present in the bag."""
        available_topics = {c.topic for c in reader.connections}
        
        # Check for old format first (25_12, 28_12, 29_12 bags)
        if '/full_data/natnet_robot_ee_pose' in available_topics:
            return TOPIC_MAP_FULL_OLD
        # Check for new full_data format
        elif '/full_data/end_effector_pose' in available_topics:
            return TOPIC_MAP_FULL
        # Fallback to throttled topics
        else:
            return TOPIC_MAP

    def read_and_sync(self, reader: AnyReader) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        # Detect which topic map to use
        self.topic_map = self._detect_topic_map(reader)
        
        # Initialize buffers for detected topic map
        self.buffers = {k: [] for k in self.topic_map.keys()}
        
        connections = [c for c in reader.connections if c.topic in self.topic_map]
        
        print("1. Reading all messages into buffers...")
        for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections)):
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # Extract Header Timestamp (preferred) or use Bag Timestamp
            if hasattr(msg, 'header'):
                ts_ns = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
            else:
                ts_ns = timestamp # Fallback if no header

            self.buffers[topic].append((ts_ns, msg))

        # Sort all buffers by timestamp (Crucial for binary search)
        for topic in self.buffers:
            self.buffers[topic].sort(key=lambda x: x[0])

        # set pivot topic as the topic with the least messages
        # drop all topics for ref images in order to find the pivot topic 
        self.buffers_pivot = {k: v for k, v in self.buffers.items() if "ref_image" not in k}
        PIVOT_TOPIC = min(self.buffers_pivot.keys(), key=lambda k: len(self.buffers_pivot[k]))

        # Use the already detected topic_map
        topic_map = self.topic_map

        print("*"*100)
        print(f"using TOPIC_MAP: {topic_map}")
        print("*"*100)
        print("2. Synchronizing samples...")
        synced_samples = {}
        
        # Track which message indices have been used for each topic (to ensure each msg used once)
        used_indices: Dict[str, set] = {k: set() for k in topic_map.keys()}
        
        # Statistics for sync failures (by topic key)
        sync_stats = {
            "total_pivot_msgs": 0,
            "sync_failures": {key: 0 for key in topic_map.values()},
            "synced_samples": 0,
        }
        
        # We iterate through the PIVOT topic to define "Timeframes"
        pivot_data = self.buffers.get(PIVOT_TOPIC, [])
        
        if not pivot_data:
            print(f"Warning: Pivot topic {PIVOT_TOPIC} is empty!")
            return {}, sync_stats

        required_keys = list(topic_map.values())
        sync_stats["total_pivot_msgs"] = len(pivot_data)

        for pivot_idx, (pivot_ts, pivot_msg) in enumerate(tqdm(pivot_data)):
            current_sample = {}
            current_sample_indices = {}  # Track indices matched for this sample
            
            # Map the pivot itself
            current_sample[topic_map[PIVOT_TOPIC]] = pivot_msg

            # Find closest match for all OTHER topics
            valid_sample = True
            failed_key = None
            for topic, key in topic_map.items():
                if topic == PIVOT_TOPIC:
                    continue
                
                result = find_closest_msg(pivot_ts, self.buffers[topic], SYNC_TOLERANCE_NS, used_indices[topic])
                if result:
                    matched_idx, matched_msg = result
                    current_sample[key] = matched_msg
                    current_sample_indices[topic] = matched_idx
                else:
                    # Packet loss / out of sync -> Discard this sample
                    valid_sample = False
                    failed_key = key
                    break
            
            # check if all required keys are present
            if valid_sample:
                for req_key in required_keys:
                    if req_key not in current_sample:
                        valid_sample = False
                        failed_key = req_key
                        break
            
            if valid_sample:
                synced_samples[pivot_ts] = current_sample
                # Mark all matched indices as used (including pivot)
                used_indices[PIVOT_TOPIC].add(pivot_idx)
                for topic, idx in current_sample_indices.items():
                    used_indices[topic].add(idx)
            else:
                if failed_key:
                    sync_stats["sync_failures"][failed_key] += 1

        sync_stats["synced_samples"] = len(synced_samples)
        return synced_samples, sync_stats

# -------------------------
# Saver
# -------------------------
def save_samples_hdf5(synced_data: Dict[int, Any], bag_path: Path, root_save_dir: Path) -> Dict[str, int]:
    """Save samples to HDF5 and return statistics about saved/rejected samples."""
    stats = {
        "total_synced": len(synced_data),
        "saved": 0,
        "rejected_marker_count": 0,
        "rejected_markers_too_close": 0,
        "rejected_self_intersection": 0,
    }
    
    if not synced_data:
        return stats

    session_folder = bag_path.parent.name
    target_dir = root_save_dir / session_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    sorted_timestamps = sorted(synced_data.keys())
    saved_count = 0
    
    print(f"Saving to {target_dir}...")
    
    for ts in tqdm(sorted_timestamps):
        sample = synced_data[ts]
        fpath = target_dir / f"sample_{saved_count}.h5"
        m_msg = sample['markers']
            # Accessing points list from Visualization Marker
        pts = [[p.x, p.y, p.z] for p in m_msg.points]
        if len(pts) != 14:
            stats["rejected_marker_count"] += 1
            continue
        
        # Check that consecutive markers are not too close to each other
        pts_arr = np.array(pts)
        marker_distances = np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)
        if np.any(marker_distances < MIN_MARKER_DISTANCE_M):
            stats["rejected_markers_too_close"] += 1
            continue
        
        # # Check for self-intersection (non-adjacent segments too close)
        # if check_self_intersection(pts_arr, SELF_INTERSECTION_THRESHOLD_M):
        #     stats["rejected_self_intersection"] += 1
        #     continue
        
        with h5py.File(fpath, 'w') as f:
            # 3. Markers

            f.create_dataset("marker_positions", data=pts)
            f.create_dataset("timestamp", data=np.int64(ts))
            
            # 1. Poses (EE and Fixed)
            for key in ['ee', 'fixed']:
                
                msg = sample[key]
                pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                ori = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
                
                f.create_dataset(f"{key}_position", data=pos)
                f.create_dataset(f"{key}_orientation", data=ori)

            # 2. Joint States
            js = sample['joint_states']
            # Save strictly what is available
            if len(js.position) > 0:
                f.create_dataset("joint_positions", data=js.position)
            if len(js.velocity) > 0:
                f.create_dataset("joint_velocities", data=js.velocity)
            if len(js.name) > 0:
                # h5py requires special handling for strings
                f.create_dataset("joint_names", data=np.array(js.name, dtype='S'))

            # 4. Images
            g_img = f.create_group("digit_images")
            # Map digit_1 -> 4
            for i in range(1, 5):
                key = f'digit_{i}'
                msg = sample[key]
                raw = np.frombuffer(msg.data, dtype=np.uint8)
                ds = g_img.create_dataset(f"img_{i-1}", data=raw)
                ds.attrs['height'] = msg.height
                ds.attrs['width'] = msg.width
                ds.attrs['encoding'] = msg.encoding
                ds.attrs['step'] = msg.step

        saved_count += 1
    
    stats["saved"] = saved_count
    return stats


def print_session_summary(session_name: str, sync_stats: Dict, save_stats: Dict) -> None:
    """Print a summary for a single session."""
    total_pivot = sync_stats["total_pivot_msgs"]
    synced = sync_stats["synced_samples"]
    sync_failures = sync_stats["sync_failures"]
    
    saved = save_stats["saved"]
    rejected_marker_count = save_stats["rejected_marker_count"]
    rejected_markers_close = save_stats["rejected_markers_too_close"]
    rejected_self_intersection = save_stats["rejected_self_intersection"]
    
    total_sync_failures = sum(sync_failures.values())
    
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")
    print(f"  Pivot messages:          {total_pivot}")
    print(f"  Synced samples:          {synced} ({100*synced/total_pivot:.1f}%)" if total_pivot > 0 else "  Synced samples:          0")
    print(f"  Sync failures:           {total_sync_failures}")
    if total_sync_failures > 0:
        for key, count in sync_failures.items():
            if count > 0:
                print(f"    - {key}: {count}")
    print(f"  Rejected (marker count):     {rejected_marker_count}")
    print(f"  Rejected (markers close):    {rejected_markers_close}")
    print(f"  Rejected (self-intersection):{rejected_self_intersection}")
    print(f"  Saved samples:           {saved}")
    print(f"{'='*60}\n")


def print_overall_summary(all_stats: List[Tuple[str, Dict, Dict]]) -> None:
    """Print overall summary across all sessions."""
    total_pivot = 0
    total_synced = 0
    total_saved = 0
    total_rejected_marker_count = 0
    total_rejected_markers_close = 0
    total_rejected_self_intersection = 0
    total_sync_failures_by_key = {}
    
    for session_name, sync_stats, save_stats in all_stats:
        total_pivot += sync_stats["total_pivot_msgs"]
        total_synced += sync_stats["synced_samples"]
        total_saved += save_stats["saved"]
        total_rejected_marker_count += save_stats["rejected_marker_count"]
        total_rejected_markers_close += save_stats["rejected_markers_too_close"]
        total_rejected_self_intersection += save_stats["rejected_self_intersection"]
        
        for key, count in sync_stats["sync_failures"].items():
            total_sync_failures_by_key[key] = total_sync_failures_by_key.get(key, 0) + count
    
    total_sync_failures = sum(total_sync_failures_by_key.values())
    
    print(f"\n{'#'*60}")
    print(f"OVERALL SUMMARY ({len(all_stats)} sessions)")
    print(f"{'#'*60}")
    print(f"  Total pivot messages:      {total_pivot}")
    print(f"  Total synced samples:      {total_synced} ({100*total_synced/total_pivot:.1f}%)" if total_pivot > 0 else "  Total synced samples:      0")
    print(f"  Total sync failures:       {total_sync_failures} ({100*total_sync_failures/total_pivot:.1f}%)" if total_pivot > 0 else "  Total sync failures:       0")
    if total_sync_failures > 0:
        print(f"    Breakdown by topic:")
        for key, count in sorted(total_sync_failures_by_key.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"      - {key}: {count}")
    print(f"  Total rejected (marker count):      {total_rejected_marker_count}")
    print(f"  Total rejected (markers close):     {total_rejected_markers_close}")
    print(f"  Total rejected (self-intersection): {total_rejected_self_intersection}")
    print(f"  Total saved samples:       {total_saved} ({100*total_saved/total_pivot:.1f}%)" if total_pivot > 0 else "  Total saved samples:       0")
    print(f"{'#'*60}\n")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Adjust these paths
    root_save_dir = Path("src/datasets/ds/train")
    bags_dir_path = Path("src/datasets/ds/train_pre")

    typestore = get_typestore(Stores.ROS2_JAZZY)
    

    mcap_files = sorted(bags_dir_path.rglob("*.mcap"))
    print(f"Found {len(mcap_files)} MCAP files.")
    
    all_session_stats = []  # List of (session_name, sync_stats, save_stats)

    for bagfile in mcap_files:
        print(f"\nProcessing bag: {bagfile.name}")
        reader_util = ApproximateBagReader()
        bag_parent = bagfile.parent
        try:
            with AnyReader([bag_parent], default_typestore=typestore) as reader:
                synced_data, sync_stats = reader_util.read_and_sync(reader)
                save_stats = save_samples_hdf5(synced_data, bagfile, root_save_dir)
                
                session_name = bag_parent.name
                all_session_stats.append((session_name, sync_stats, save_stats))
                print_session_summary(session_name, sync_stats, save_stats)
                
        except Exception as e:
            print(f"Error reading bag {bagfile}: {e}")
            # import traceback
            # traceback.print_exc()
    
    # Print overall summary at the end
    if all_session_stats:
        print_overall_summary(all_session_stats)