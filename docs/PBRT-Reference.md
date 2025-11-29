# PBRT Reference 

## Install [TEV Display server](github.com/Tom94/tev) and play around with `pbrt`

`pbrt` uses `tev` to display its image during rendering (as an alternative to `glfw`). If both `tev` and `pbrt` are on the path, then you can execute (Powershell)

- windows

    ```powershell
    Start-Process -FilePath "tev" -ArgumentList "--hostname","127.0.0.1:14158"
    ```

- linux

  '''sh
  tev --hostname 127.0.0.1:14158 &
    ```

Once you have a `tev` server up and running,

```sh
# (GPU version)
pbrt --gpu --log-level verbose --display-server 127.0.0.1:14158 .\villa-daylight.pbrt
# (CPU version)
pbrt --wavefront --log-level verbose --display-server 127.0.0.1:14158 .\villa-daylight.pbrt
```

Alternatively, `pbrt` can also display to a native, `glfw` based window with the `--interactive` option.
(one of `--interactive` and `--display-server <addr:port>` can be used, not both). It's laggy so I don't reccomend it.
