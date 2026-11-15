#!/usr/bin/env python3
"""Read a PS5 DualSense with pygame and stream its state to the ROS machine via UDP.

Runs on the machine the controller is attached to, w/o ROS

    pip install pygame
    python3 dualsense_sender.py <VM_IP>            # stream to the ROS node
    python3 dualsense_sender.py --probe            # show button/axis/hat numbers

Packet (JSON, ~30 per second): {"axes": [...], "buttons": [0/1, ...], "hats": [[x, y], ...]}
"""
import argparse
import json
import os
import socket
import time
import pygame  # noqa: E402

# Keep receiving joystick events although this terminal has no window / focus.
# Must be set BEFORE pygame is imported.
os.environ.setdefault('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')


def open_first_joystick():
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        print(f'Controller: {js.get_name()} ({js.get_numaxes()} axes, '
              f'{js.get_numbuttons()} buttons, {js.get_numhats()} hats)')
        return js
    return None


def update_joystick(js):
    """Handle hot-plugging; returns the (possibly new) joystick or None."""
    for event in pygame.event.get():
        if event.type == pygame.JOYDEVICEADDED and js is None:
            js = pygame.joystick.Joystick(event.device_index)
            print(f'Controller connected: {js.get_name()}')
        elif event.type == pygame.JOYDEVICEREMOVED and js is not None:
            print('Controller disconnected')
            js = None
    return js


def probe():
    js = open_first_joystick()
    print('Move sticks / press buttons. CTRL-C to quit.')
    clock = pygame.time.Clock()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.JOYDEVICEADDED and js is None:
                js = pygame.joystick.Joystick(e.device_index)
                print(f'Controller connected: {js.get_name()}')
            elif e.type == pygame.JOYBUTTONDOWN:
                print('button', e.button)
            elif e.type == pygame.JOYAXISMOTION and abs(e.value) > 0.5:
                print('axis', e.axis, round(e.value, 2))
            elif e.type == pygame.JOYHATMOTION:
                print('hat', e.hat, e.value)
        clock.tick(60)


def stream(host, port, rate):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    js = open_first_joystick()
    period = 1.0 / rate
    print(f'Streaming to {host}:{port} at {rate} Hz. CTRL-C to quit.')
    while True:
        js = update_joystick(js)
        if js is not None:
            state = {
                'axes': [round(js.get_axis(i), 3) for i in range(js.get_numaxes())],
                'buttons': [js.get_button(i) for i in range(js.get_numbuttons())],
                'hats': [list(js.get_hat(i)) for i in range(js.get_numhats())],
            }
            sock.sendto(json.dumps(state).encode(), (host, port))
            print(f"Sending state: {state}")
        # no controller -> send nothing; the receiver's watchdog stops the robot
        time.sleep(period)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('host', nargs='?', help='IP/hostname of the machine running the ROS node')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--rate', type=float, default=30.0, help='packets per second')
    parser.add_argument('--probe', action='store_true', help='print button/axis/hat numbers')
    args = parser.parse_args()

    if not args.probe and not args.host:
        parser.error('host is required unless --probe is used')

    pygame.init()
    pygame.joystick.init()
    try:
        if args.probe:
            probe()
        else:
            stream(args.host, args.port, args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()