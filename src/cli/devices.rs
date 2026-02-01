use anyhow::Result;
use wgpu::{Backends, Instance};

pub fn list_devices() -> Result<Vec<Device>> {
    let descriptor = wgpu::InstanceDescriptor::default();
    let instance = Instance::new(&descriptor);
    let mut devices = Vec::new();

    let adapters = instance.enumerate_adapters(Backends::all());
    for adapter in adapters {
        let info = adapter.get_info();
        let backend_type = format!("{:?}", info.backend);
        devices.push(Device {
            name: info.name.clone(),
            backend: backend_type,
        });
    }

    Ok(devices)
}

pub fn select_default_device(devices: &[Device]) -> Option<&Device> {
    // Simple selection: pick the first one
    devices.first()
}

#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub backend: String,
}

pub fn list_devices_impl() {
    match list_devices() {
        Ok(devices) => {
            println!("Available GPU devices:");
            if devices.is_empty() {
                println!("  No compatible GPU devices found");
            } else {
                for (i, device) in devices.iter().enumerate() {
                    println!("  [{}. {}] ({})", i + 1, device.name, device.backend);
                }
            }
        }
        Err(e) => {
            eprintln!("Error listing devices: {}", e);
        }
    }
}
