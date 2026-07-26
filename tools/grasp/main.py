#!/usr/bin/env python3
"""
grasp 抓取任务主入口 — 8-phase 流程
用法：python3 main.py [--config config.yaml] [--mode pc|robot] [--zone A]
"""
import sys
import os
import argparse
import logging
import time
import yaml
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ArmController        import ArmController, SAFE_ANGLE_3, SAFE_ANGLE_4, SAFE_ANGLE_5
from utils.BlockDetection       import BlockDetection
from utils.TargetTracker        import TargetTracker
from utils.InspectionMemory     import InspectionMemory
from utils.DogAlignInterface    import DogAlignInterface
from utils.RobotSignalInterface import RobotSignalInterface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("grasp_run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ─────────────────────────── 辅助 ────────────────────────────────────────── #

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pc_wait(prompt: str) -> None:
    """pc 模式下等待用户按回车。"""
    try:
        input(prompt)
    except EOFError:
        pass   # 非交互环境（管道/测试）直接跳过

# ─────────────────────────── phase_0 ─────────────────────────────────────── #

def phase_0_init(cfg: dict, mode: str) -> dict | None:
    """初始化所有模块，返回共享上下文 ctx；失败返回 None。"""
    logger.info("=== phase_0: 初始化 (mode=%s) ===", mode)
    try:
        arm = ArmController(
            device=cfg["hardware"]["arm_serial_port"],
            cfg={**cfg["arm"], "arm_serial_baud": cfg["hardware"]["arm_serial_baud"]},
        )
        detector = BlockDetection({**cfg["detection"]})
        memory   = InspectionMemory(default_zone=cfg["inspection"]["default_zone"])
        dog_align  = DogAlignInterface(mode=mode)
        robot_sig  = RobotSignalInterface(mode=mode)

        cfg_g   = cfg["grasp"]
        tracker = TargetTracker(
            avg_window=int(cfg_g["distance_avg_window"]),
            lost_frames_max=int(cfg_g["lost_frames_max"]),
        )

        cam_device = cfg["hardware"]["arm_cam_device"]
        arm_cam = cv2.VideoCapture(cam_device, cv2.CAP_V4L2)
        if not arm_cam.isOpened():
            logger.error("机械臂摄像头打开失败: %s", cam_device)
            arm.finalize()
            return None

        logger.info("初始化完成。摄像头: %s  串口: %s",
                    cam_device, cfg["hardware"]["arm_serial_port"])
        return {
            "cfg":       cfg,
            "mode":      mode,
            "arm":       arm,
            "detector":  detector,
            "tracker":   tracker,
            "memory":    memory,
            "dog_align": dog_align,
            "robot_sig": robot_sig,
            "arm_cam":   arm_cam,
        }
    except Exception as e:
        logger.exception("初始化失败: %s", e)
        return None

# ─────────────────────────── phase_1 ─────────────────────────────────────── #

def phase_1_standby(ctx: dict) -> bool:
    """机械臂进入 mode=2 相机初始位姿，等待机器狗到达 place1 停稳。"""
    logger.info("=== phase_1: 待命 ===")
    arm = ctx["arm"]
    arm.set_pose(0)
    arm.set_pose(2)
    logger.info("机械臂就绪，等待机器狗停稳...")

    ok = ctx["robot_sig"].wait_start()
    if not ok:
        logger.error("等待停稳信号超时")
        return False

    if ctx["mode"] == "pc":
        _pc_wait("确认机器狗已到达 place1 停稳，按回车继续...")
    return True

# ─────────────────────────── phase_2 ─────────────────────────────────────── #

def phase_2_detect(ctx: dict) -> dict | None:
    """
    多帧检测，TargetTracker 滑动均值稳定后返回稳定目标读数。
    返回 dict（与 detect() 结构相同）或 None（超时）。
    """
    logger.info("=== phase_2: 视觉识别 ===")
    detector = ctx["detector"]
    tracker  = ctx["tracker"]
    arm_cam  = ctx["arm_cam"]
    timeout  = float(ctx["cfg"]["grasp"]["detect_timeout"])
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        ret, frame = arm_cam.read()
        if not ret:
            logger.warning("摄像头读帧失败，跳过")
            continue

        candidates = detector.detect_all(frame)
        tracker.update(candidates)

        # 可视化调试（只显示最近候选）
        vis_result = candidates[0] if candidates else None
        vis = detector.visualize(frame.copy(), vis_result)
        cv2.imshow("arm_cam", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

        stable = tracker.get_stable_target()
        if stable is not None:
            logger.info("目标锁定稳定: dist=%.1fmm offset_x=%d",
                        stable["distance_mm"], stable["center_offset_x"])
            return stable

        if candidates:
            logger.debug("候选 %d 个，窗口未满，继续采样", len(candidates))
        else:
            logger.debug("未检测到色块")

    logger.error("phase_2 超时 (%.1fs)，未检测到稳定色块", timeout)
    return None

# ─────────────────────────── phase_3 ─────────────────────────────────────── #

def phase_3_align(ctx: dict, stable: dict) -> bool:
    """
    根据 X_cam 偏移通知机器狗横向调整到 place2，直到对齐。
    """
    logger.info("=== phase_3: 横向对齐 ===")
    cfg_g     = ctx["cfg"]["grasp"]
    detector  = ctx["detector"]
    tracker   = ctx["tracker"]
    arm_cam   = ctx["arm_cam"]
    dog_align = ctx["dog_align"]
    thr_mm    = float(cfg_g["align_offset_threshold_mm"])
    timeout   = float(cfg_g["detect_timeout"])

    # X_cam 从 pos_3d 取（单位 mm）
    X_cam = stable["pos_3d"][0]
    max_rounds = 5   # 最多对齐 5 轮，防止死循环

    for round_i in range(max_rounds):
        if abs(X_cam) <= thr_mm:
            logger.info("横向已对齐: X_cam=%.1fmm (阈值=%.1fmm)", X_cam, thr_mm)
            return True

        dog_align.send_align(X_cam)
        logger.info("发送横向调整: %.1fmm", X_cam)

        if ctx["mode"] == "pc":
            _pc_wait(f"[pc] 请手动调整机器狗横向 {X_cam:.1f}mm，完成后按回车...")
        else:
            ok = dog_align.wait_aligned(timeout=15.0)
            if not ok:
                logger.error("等待对齐完成超时")
                return False

        # 调整后重新采样
        tracker.__init__(
            avg_window=int(cfg_g["distance_avg_window"]),
            lost_frames_max=int(cfg_g["lost_frames_max"]),
        )
        new_stable = phase_2_detect(ctx)
        if new_stable is None:
            logger.error("对齐后重新识别失败")
            return False
        X_cam = new_stable["pos_3d"][0]
        stable.update(new_stable)

    logger.error("横向对齐超过最大轮次 (%d)，仍未对齐 X_cam=%.1fmm", max_rounds, X_cam)
    return False

# ─────────────────────────── phase_4 ─────────────────────────────────────── #

def phase_4_approach_grasp(ctx: dict, stable: dict) -> bool:
    """
    两步接近抓取。
    步骤1: 移到 dis_safe，下降到 h_object（单次 grap，允许弧线）
    步骤2: 以固定步长 step_mm 等高插值前进到 dis_target，三轴同步写保持恒高
    步骤3: grasp_with_verify 夹取
    """
    logger.info("=== phase_4: 接近与抓取 ===")
    arm   = ctx["arm"]
    cfg_g = ctx["cfg"]["grasp"]

    cv2.destroyAllWindows()

    clearance   = float(cfg_g["approach_clearance_mm"])
    h_object    = float(cfg_g["h_object"])
    dist_offset = float(cfg_g.get("distance_offset_mm", 0.0))
    step_mm     = float(cfg_g.get("approach_step_mm", 5.0))

    X_cam, Y_cam, Z_cam = stable["pos_3d"]
    dis_target = Y_cam + dist_offset
    dis_safe   = dis_target - clearance

    logger.info("物块坐标（相机系）: X=%.1fmm  Y=%.1fmm  Z=%.1fmm", X_cam, Y_cam, Z_cam)
    logger.info("IK 输入: dis_safe=%.1fmm → dis=%.1fmm, h=%.1fmm, step=%.1fmm",
                dis_safe, dis_target, h_object, step_mm)

    from utils.RobotArm.three_Inverse_kinematics import Arm as IKArm
    import math

    if dis_safe < 30.0 or dis_safe >= dis_target:
        # 距离不足，跳过步骤1，直接从当前位姿开始等高前进
        logger.info("步骤1跳过（dis_safe=%.1fmm 不足），从当前位置直接等高前进", dis_safe)
        dis_safe = dis_target - step_mm   # 让步骤2至少走一步到 dis_target
    else:
        # 步骤 1：移到安全距离并下降到目标高度（允许弧线运动）
        logger.info("步骤1: dis=%.1fmm  h=%.1fmm", dis_safe, h_object)
        ok = arm.grap(dis_safe, h_object)
        if not ok:
            logger.error("步骤1 IK 解超出范围 (dis=%.1f h=%.1f)", dis_safe, h_object)
            return False
        a3, a4, a5 = IKArm(dis_safe, h_object)
        arm.wait_for_position({3: a3, 4: a4, 5: a5})

    # 步骤 2：等高插值前进，三轴同步写（无论步骤1是否执行都运行）
    n_steps = max(1, math.ceil((dis_target - dis_safe) / step_mm))
    logger.info("步骤2: 等高前进 %.1f→%.1fmm，%d步", dis_safe, dis_target, n_steps)
    ph  = arm.packetHandler
    spd = arm._speed
    acc = arm._acc

    for i in range(1, n_steps + 1):
        dis_i = dis_safe + (dis_target - dis_safe) * i / n_steps
        a3_i, a4_i, a5_i = IKArm(dis_i, h_object)
        if not (SAFE_ANGLE_3[0] <= a3_i <= SAFE_ANGLE_3[1] and
                SAFE_ANGLE_4[0] <= a4_i <= SAFE_ANGLE_4[1] and
                SAFE_ANGLE_5[0] <= a5_i <= SAFE_ANGLE_5[1]):
            logger.error("插值步 %d/%d IK 超出安全范围 (dis=%.1f)", i, n_steps, dis_i)
            return False
        ph.WritePosEx(3, a3_i, spd, acc)
        ph.WritePosEx(4, a4_i, spd, acc)
        ph.WritePosEx(5, a5_i, spd, acc)
        arm.wait_for_position({3: a3_i, 4: a4_i, 5: a5_i})
        logger.debug("插值步 %d/%d dis=%.1fmm", i, n_steps, dis_i)

    # 步骤 3：执行夹取
    logger.info("步骤3: 夹取 dis=%.1fmm  h=%.1fmm", dis_target, h_object)
    success = arm.grasp_with_verify(dis=dis_target, height=h_object)
    if success:
        logger.info("抓取成功")
    else:
        logger.error("抓取失败（已重试 %s 次）", cfg_g.get("grasp_retry_max", 3))
    return success

# ─────────────────────────── phase_5 ─────────────────────────────────────── #

def phase_5_transport(ctx: dict) -> bool:
    """进入运输姿态，保持夹爪不松开。"""
    logger.info("=== phase_5: 运输姿态 ===")
    try:
        ctx["arm"].set_pose(3, keep_gripper=True)
        return True
    except Exception as e:
        logger.error("运输姿态失败: %s", e)
        return False

# ─────────────────────────── phase_6 ─────────────────────────────────────── #

def phase_6_place(ctx: dict) -> bool:
    """等放置触发信号，移到放置区，松开夹爪。"""
    logger.info("=== phase_6: 放置 ===")
    arm    = ctx["arm"]
    memory = ctx["memory"]
    cfg_p  = ctx["cfg"]["placement"]

    zone = memory.get_zone()
    zone_cfg = cfg_p["zones"].get(zone)
    if zone_cfg is None:
        logger.error("未知放置区: %s", zone)
        return False

    ok = ctx["robot_sig"].wait_place(zone=zone)
    if not ok:
        logger.error("等待放置信号超时")
        return False
    if ctx["mode"] == "pc":
        _pc_wait(f"确认机器狗已到达 place2 放置位（区={zone}），按回车继续...")

    dis    = float(zone_cfg["dis"])
    height = float(zone_cfg["height"])
    logger.info("放置到 %s 区 (dis=%.1fmm, height=%.1fmm)", zone, dis, height)

    try:
        from utils.RobotArm.three_Inverse_kinematics import Arm as IKArm
        ok = arm.grap(dis, height, keep_gripper=True)
        if not ok:
            logger.error("放置 IK 解超出范围")
            return False
        a3, a4, a5 = IKArm(dis, height)
        arm.wait_for_position({3: a3, 4: a4, 5: a5})
        time.sleep(float(cfg_p.get("lower_timeout", 2.0)))
        arm.open_gripper()
        logger.info("已放置，夹爪已张开")
        return True
    except Exception as e:
        logger.error("放置失败: %s", e)
        return False

# ─────────────────────────── phase_7 ─────────────────────────────────────── #

def phase_7_home(ctx: dict) -> None:
    """归位 mode=0，释放资源。"""
    logger.info("=== phase_7: 归位 ===")
    try:
        ctx["arm"].set_pose(0)
    except Exception as e:
        logger.warning("归位时异常: %s", e)
    finally:
        ctx["arm_cam"].release()
        cv2.destroyAllWindows()
        ctx["arm"].finalize()
        logger.info("资源已释放")

# ─────────────────────────── 主流程 ──────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="grasp 抓取任务")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--mode", default=None,
                        help="运行模式 pc|robot（覆盖 config.yaml 中的 runtime.mode）")
    parser.add_argument("--zone", default=None,
                        help="手动指定放置区，如 --zone B")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    mode = args.mode or cfg.get("runtime", {}).get("mode", "pc")
    logger.info("配置加载完成: %s  mode=%s", args.config, mode)

    ctx = phase_0_init(cfg, mode)
    if ctx is None:
        sys.exit(1)

    if args.zone:
        ctx["memory"].set_zone(args.zone.upper())
        logger.info("手动设置放置区: %s", args.zone.upper())

    try:
        if not phase_1_standby(ctx):
            sys.exit(1)

        stable = phase_2_detect(ctx)
        if stable is None:
            sys.exit(1)

        if not phase_3_align(ctx, stable):
            sys.exit(1)

        if not phase_4_approach_grasp(ctx, stable):
            sys.exit(1)

        if not phase_5_transport(ctx):
            sys.exit(1)

        if not phase_6_place(ctx):
            sys.exit(1)

        logger.info("任务完成！")

    except KeyboardInterrupt:
        logger.warning("用户中断，执行安全归位")
    finally:
        phase_7_home(ctx)


if __name__ == "__main__":
    main()
