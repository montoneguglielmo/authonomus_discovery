"""
JacoThreeFingerGripper with per-finger MuJoCo touch sensors added programmatically.

Injects <site> elements into each fingertip body and <touch> sensor elements into
the <sensor> block after the parent class has loaded the XML and applied name prefixes.
Registers itself with robosuite via @register_gripper so it can be selected by name.
"""
import xml.etree.ElementTree as ET

from robosuite.models.grippers.jaco_three_finger_gripper import JacoThreeFingerGripper
from robosuite.models.grippers import register_gripper
from robosuite.utils.mjcf_utils import find_elements


# Raw (un-prefixed) fingertip body names and the site position matching each
# *_tip_collision geom center (from jaco_three_finger_gripper.xml).
_FINGER_CONFIGS = {
    "thumb_distal": {"pos": "0 -0.003 0.021", "sensor": "touch_thumb"},
    "index_distal": {"pos": "0  0.003 0.021", "sensor": "touch_index"},
    "pinky_distal":  {"pos": "0  0.003 0.021", "sensor": "touch_pinky"},
}


@register_gripper
class JacoThreeFingerTouchGripper(JacoThreeFingerGripper):
    """
    JacoThreeFingerGripper with one MuJoCo touch sensor per fingertip.

    After super().__init__() finishes (which loads the XML, sorts elements into
    self._sites / self._sensors with raw names, and applies self.naming_prefix to
    all XML element attributes via add_prefix), we inject:
      - A <site> element into each *_distal body (already prefixed in the XML tree)
      - A <touch> sensor element into self.sensor
      - Raw names into self._sites and self._sensors so .sites/.sensors stay consistent

    Touch sensor values are read via sim.data.sensordata; use _get_touch_sensor_data()
    on the environment (defined in RandomObjectsEnv) for convenient access.
    """

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._inject_touch_sensors()

    def _inject_touch_sensors(self):
        """
        Inject touch sensor sites and sensors into the already-prefixed XML tree.

        self.naming_prefix (e.g. "0_") has already been applied to the XML by
        add_prefix, so new element attribute values must use the prefixed names.
        self._sites / self._sensors store raw names (correct_naming() prefixes them),
        so we append raw names there.
        """
        pfx = self.naming_prefix  # e.g. "0_" for idn=0

        for raw_body, cfg in _FINGER_CONFIGS.items():
            raw_sensor = cfg["sensor"]
            raw_site = raw_sensor + "_site"

            prefixed_body   = pfx + raw_body
            prefixed_site   = pfx + raw_site
            prefixed_sensor = pfx + raw_sensor

            # Find the distal body in the already-prefixed worldbody tree
            body_el = find_elements(
                root=self.worldbody,
                tags="body",
                attribs={"name": prefixed_body},
                return_first=True,
            )
            if body_el is None:
                raise RuntimeError(
                    f"[JacoThreeFingerTouchGripper] Could not find body "
                    f"'{prefixed_body}' in gripper worldbody. "
                    f"Check that the parent class XML was loaded correctly."
                )

            # Add an invisible site on the fingertip body.
            # The site acts as the reference frame for the touch sensor.
            # Size matches the *_tip_collision geom for conceptual consistency.
            site_el = ET.Element("site", {
                "name":  prefixed_site,
                "pos":   cfg["pos"],
                "size":  "0.01 0.005 0.02",
                "type":  "box",
                "rgba":  "0 1 0 0",   # fully transparent
                "group": "1",
            })
            body_el.append(site_el)

            # Add the touch sensor referencing the site above.
            # MuJoCo touch sensors return the total normal contact force (scalar N)
            # on the body containing the site.
            touch_el = ET.Element("touch", {
                "name": prefixed_sensor,
                "site": prefixed_site,
            })
            self.sensor.append(touch_el)

            # Register raw names in the model's internal tracking lists.
            # The .sites and .sensors properties call correct_naming() which
            # prepends naming_prefix, producing the same prefixed names we
            # wrote into the XML attributes above.
            self._sites.append(raw_site)
            self._sensors.append(raw_sensor)
