# Vendored LBFGS++

- Upstream: https://github.com/yixuan/LBFGSpp
- Version: v0.4.0
- Commit: `c524a407fb85b74807f53de5a3ca2ddbcc164e54`
- Archive: https://github.com/yixuan/LBFGSpp/archive/c524a407fb85b74807f53de5a3ca2ddbcc164e54.tar.gz
- Archive SHA-256: `8fbb7037c4e0bf0d83ced74463092154038bc0bbb43f77fe38f2bc7c3414c2d8`
- License: [MIT](LICENSE.md)

The complete upstream `include/` directory, `LICENSE.md`, and `AUTHORS.md` are
copied without modification. CMake uses these local headers as system includes;
LBFGS++ is not downloaded during configuration or compilation.

To update, obtain the desired upstream archive, verify its checksum, replace the
headers and upstream attribution files, and update the version information here.
Run the C++ tests, scikit-learn reference tests, and clang-tidy after updating.
