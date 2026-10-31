# RealFact build. Apple clang has no bundled OpenMP runtime, so we use the
# prebuilt libomp in third_party/libomp (LLVM 17, from mac.r-project.org/openmp).
# On Linux/GCC: make OMP_CFLAGS=-fopenmp OMP_LIBS=-fopenmp SHARED=-shared EXT=so

CXX      ?= clang++
OMP_DIR  := third_party/libomp
EXT      ?= dylib
SHARED   ?= -dynamiclib
CXXFLAGS := -O3 -std=c++17 -fPIC -ffp-contract=off -Wall
OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OMP_DIR)/include
OMP_LIBS   ?= -L$(OMP_DIR)/lib -lomp -Wl,-rpath,@loader_path/../$(OMP_DIR)/lib

BUILD := build

all: $(BUILD)/librealfact_omp.$(EXT) $(BUILD)/librealfact_serial.$(EXT)

$(OMP_DIR)/lib/.fixed:
	install_name_tool -id @rpath/libomp.dylib $(OMP_DIR)/lib/libomp.dylib 2>/dev/null
	codesign -f -s - $(OMP_DIR)/lib/libomp.dylib
	touch $@

$(BUILD)/librealfact_omp.$(EXT): engine/realfact_engine.cpp $(OMP_DIR)/lib/.fixed
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $(SHARED) $< -o $@ $(OMP_LIBS)

# Serial reference T1: same source, compiled WITHOUT OpenMP (methodology §III-A).
$(BUILD)/librealfact_serial.$(EXT): engine/realfact_engine.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(SHARED) $< -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean

# LaTeX report (needs tectonic or pdflatex on PATH)
report: report/realfact_report.pdf
report/realfact_report.pdf: report/realfact_report.tex
	cd report && (test -x ../third_party/tectonic/tectonic && ../third_party/tectonic/tectonic realfact_report.tex || (pdflatex -interaction=nonstopmode realfact_report.tex && pdflatex -interaction=nonstopmode realfact_report.tex))
.PHONY: report
