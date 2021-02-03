from conans import CMake, ConanFile, tools
from conans.errors import ConanInvalidConfiguration

class TensorFlowLiteTest(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"

    @property
    def _android(self):
        return self.settings.os == "Android"

    @property
    def _linux(self):
        return self.settings.os == "Linux"

    def _req_add(self, req, options={}):
        self.requires.add(req)
        package = req.split("/")[0]
        for option, value in options.items():
            setattr(self.options[package], option, value)

    def _req_override(self, req):
        self.requires.add(req, override=True)

    def build_requirements(self):
        yield

    def requirements(self):
        #if self._linux:
        #    raise Exception("Linux not suported yet")

        self._req_add("tensorflow_lite/2.5@wrnch/stable")

    def config_options(self):
        yield

    def build(self):
        generator = {
            "Linux": "Ninja",
            "Android": "Unix Makefiles",
        }[str(self.settings.os)]

        cmake = CMake(self, generator=generator, toolset=None)
        cmake.configure()
        cmake.build()
