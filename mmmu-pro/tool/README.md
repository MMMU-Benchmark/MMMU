# Screenshot Generator

This tool is designed to assist the open-source community in generating datasets similar to the MMMU-Pro benchmark. The goal is to advance research in the unification of modalities within multimodal models.

## Usage

1. **Prepare Your Data**: Before running the script, ensure your data is ready and placed in the `data.jsonl` file within the `tool` directory. This JSONL file should contain the following keys:
   - `question`: The question text, which can include image slots in the format `<image x>`, where `x` is the image index.
   - `options`: The answer options, which can also include image slots in the format `<image x>`, where `x` is the image index.
   - `image_x`: Represents the path to images appearing in the question, where `x` is the index. The path should be a relative path under the static directory.

2. **Set Up Chromedriver**: In the `tool.py` file, ensure you set the path to your `chromedriver` at line `112`.

3. **Run the Script**:
   ```bash
   cd tool
   python screenshot_generator.py
   ```

   After running the script, images will be saved sequentially in the `output` folder located in the same directory.

## Additional Information

- The tool uses Flask to serve a web interface for viewing and editing the questions and options.
- Screenshots of the pages are taken using Selenium with a headless Chrome browser.

Feel free to contribute to the project or report any issues you encounter!
