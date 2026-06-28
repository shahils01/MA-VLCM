#!/usr/bin/env python3
"""Plot live MA-VLCM predictions against live task progress."""

import argparse
import json
from collections import deque

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime environment dependent
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from std_msgs.msg import String

    _ROS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    rclpy = None
    ExternalShutdownException = None
    Node = object
    String = None
    _ROS_IMPORT_ERROR = exc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_topic", default="/fleet_vlcm/vlcm_prediction")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--refresh_hz", type=float, default=5.0)
    return parser.parse_args()


class Tb3LiveMonitorNode(Node):
    def __init__(self, args):
        super().__init__("tb3_vlcm_live_monitor")
        self.args = args
        maxlen = max(1, int(args.window_size))
        self.steps = deque(maxlen=maxlen)
        self.predictions = deque(maxlen=maxlen)
        self.targets = deque(maxlen=maxlen)
        self.errors = deque(maxlen=maxlen)
        self.episode_id = ""
        self.create_subscription(String, args.prediction_topic, self.on_prediction, 10)

        plt.ion()
        self.fig, (self.ax_values, self.ax_error) = plt.subplots(2, 1, sharex=True)
        self.pred_line, = self.ax_values.plot([], [], label="MA-VLCM prediction")
        self.target_line, = self.ax_values.plot([], [], label="Progress target")
        self.error_line, = self.ax_error.plot([], [], color="tab:red", label="Absolute error")
        self.ax_values.set_ylabel("Progress")
        self.ax_error.set_ylabel("Abs error")
        self.ax_error.set_xlabel("Live step")
        self.ax_values.legend(loc="upper left")
        self.ax_error.legend(loc="upper left")
        self.fig.tight_layout()
        self.fig.show()

        period = 1.0 / max(0.1, float(args.refresh_hz))
        self.create_timer(period, self.redraw)
        self.get_logger().info(f"Listening for MA-VLCM predictions on {args.prediction_topic}")

    def on_prediction(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring malformed prediction JSON.")
            return
        new_episode_id = payload.get("episode_id", "")
        if new_episode_id and new_episode_id != self.episode_id:
            self.get_logger().info(f"New episode detected: {new_episode_id}. Clearing plot data.")
            self.episode_id = new_episode_id
            self.steps.clear()
            self.predictions.clear()
            self.targets.clear()
            self.errors.clear()

        self.steps.append(int(payload.get("step", len(self.steps))))
        self.predictions.append(float(payload.get("prediction", 0.0)))
        target = payload.get(
            "progress_target",
            payload.get("target", payload.get("cumulative_reward", 0.0)),
        )
        self.targets.append(float(target))
        self.errors.append(float(payload.get("abs_error", abs(self.predictions[-1] - self.targets[-1]))))

    def redraw(self):
        if not self.steps:
            plt.pause(0.001)
            return
        x = list(self.steps)
        self.pred_line.set_data(x, list(self.predictions))
        self.target_line.set_data(x, list(self.targets))
        self.error_line.set_data(x, list(self.errors))
        self.ax_values.relim()
        self.ax_values.autoscale_view()
        self.ax_error.relim()
        self.ax_error.autoscale_view()
        self.fig.suptitle(f"MA-VLCM Live Critic - {self.episode_id}")
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    if _ROS_IMPORT_ERROR is not None:
        raise RuntimeError("tb3_live_monitor requires a ROS 2 Python environment.") from _ROS_IMPORT_ERROR
    if _MATPLOTLIB_IMPORT_ERROR is not None:
        raise RuntimeError("tb3_live_monitor requires matplotlib.") from _MATPLOTLIB_IMPORT_ERROR

    args = parse_args()
    rclpy.init()
    node = Tb3LiveMonitorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
