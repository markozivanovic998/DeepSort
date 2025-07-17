import yaml

def load_config(app_config_path='config/app.yaml', rtsp_config_path='config/rtsp.yaml'):
    """Učitava konfiguracione fajlove za aplikaciju i RTSP."""
    with open(app_config_path, 'r') as f:
        app_config = yaml.safe_load(f)
        
    with open(rtsp_config_path, 'r') as f:
        rtsp_config = yaml.safe_load(f)

    rtsp_details = rtsp_config['rtsp']
    rtsp_url = (f"rtsp://{rtsp_details['username']}:{rtsp_details['password']}@"
                f"{rtsp_details['ip']}:{rtsp_details['port']}{rtsp_details['path']}")

    return app_config, rtsp_url