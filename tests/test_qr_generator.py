import pytest
from PIL import Image
import os
import tempfile
from utils.qr_generator import QRGenerator
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H

@pytest.fixture
def qr_generator():
    """Create a QR generator instance."""
    return QRGenerator()

@pytest.fixture
def test_data():
    """Sample data for QR code generation."""
    return "https://example.com"

@pytest.fixture
def test_logo():
    """Create a temporary test logo."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('RGB', (100, 100), color='red')
        img.save(f.name)
        yield f.name
    os.unlink(f.name)

def test_qr_generator_initialization():
    """Test QR generator initialization with different parameters."""
    # Test default initialization
    qr = QRGenerator()
    assert qr.version == 1
    assert qr.error_correction == ERROR_CORRECT_L
    assert qr.box_size == 10
    assert qr.border == 4
    assert qr.fill_color == "black"
    assert qr.back_color == "white"
    
    # Test custom initialization
    qr = QRGenerator(
        version=2,
        error_correction=ERROR_CORRECT_H,
        box_size=20,
        border=2,
        fill_color="blue",
        back_color="yellow"
    )
    assert qr.version == 2
    assert qr.error_correction == ERROR_CORRECT_H
    assert qr.box_size == 20
    assert qr.border == 2
    assert qr.fill_color == "blue"
    assert qr.back_color == "yellow"

def test_generate_basic_qr(qr_generator, test_data):
    """Test basic QR code generation."""
    qr_image = qr_generator.generate(test_data)
    
    assert isinstance(qr_image, Image.Image)
    assert qr_image.mode == 'RGB'
    assert qr_image.size[0] > 0
    assert qr_image.size[1] > 0

def test_generate_qr_with_logo(qr_generator, test_data, test_logo):
    """Test QR code generation with logo."""
    qr_image = qr_generator.generate(test_data, logo_path=test_logo)
    
    assert isinstance(qr_image, Image.Image)
    assert qr_image.mode == 'RGB'
    assert qr_image.size[0] > 0
    assert qr_image.size[1] > 0

def test_generate_qr_with_custom_logo_size(qr_generator, test_data, test_logo):
    """Test QR code generation with custom logo size."""
    logo_size = (50, 50)
    qr_image = qr_generator.generate(test_data, logo_path=test_logo, logo_size=logo_size)
    
    assert isinstance(qr_image, Image.Image)
    assert qr_image.mode == 'RGB'
    assert qr_image.size[0] > 0
    assert qr_image.size[1] > 0

def test_generate_base64(qr_generator, test_data):
    """Test base64 QR code generation."""
    base64_str = qr_generator.generate_base64(test_data)
    
    assert isinstance(base64_str, str)
    assert len(base64_str) > 0
    assert base64_str.startswith('iVBORw0KGgo')  # PNG base64 header

def test_generate_file(qr_generator, test_data):
    """Test QR code file generation."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name
    
    try:
        qr_generator.generate_file(test_data, output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Verify the file is a valid image
        img = Image.open(output_path)
        assert img.mode == 'RGB'
        assert img.size[0] > 0
        assert img.size[1] > 0
    finally:
        os.unlink(output_path)

def test_validate_data(qr_generator):
    """Test data validation."""
    # Test valid data
    assert qr_generator.validate_data("https://example.com")
    assert qr_generator.validate_data("Hello, World!")
    assert qr_generator.validate_data("1234567890")
    
    # Test invalid data (too long)
    long_data = "x" * 3000
    assert not qr_generator.validate_data(long_data)

def test_get_qr_size(qr_generator, test_data):
    """Test QR code size calculation."""
    size = qr_generator.get_qr_size(test_data)
    
    assert isinstance(size, tuple)
    assert len(size) == 2
    assert size[0] > 0
    assert size[1] > 0

def test_error_handling(qr_generator):
    """Test error handling."""
    # Test invalid logo path
    with pytest.raises(Exception):
        qr_generator.generate("test", logo_path="nonexistent.png")
    
    # Test invalid output path
    with pytest.raises(Exception):
        qr_generator.generate_file("test", "/invalid/path/qr.png")
    
    # Test invalid data type
    with pytest.raises(Exception):
        qr_generator.generate(123)  # type: ignore

def test_different_error_correction_levels():
    """Test QR code generation with different error correction levels."""
    data = "https://example.com"
    
    # Test all error correction levels
    for error_correction in [ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H]:
        qr = QRGenerator(error_correction=error_correction)
        qr_image = qr.generate(data)
        
        assert isinstance(qr_image, Image.Image)
        assert qr_image.mode == 'RGB'
        assert qr_image.size[0] > 0
        assert qr_image.size[1] > 0

def test_different_versions():
    """Test QR code generation with different versions."""
    data = "https://example.com"
    
    # Test different versions
    for version in [1, 2, 3, 4]:
        qr = QRGenerator(version=version)
        qr_image = qr.generate(data)
        
        assert isinstance(qr_image, Image.Image)
        assert qr_image.mode == 'RGB'
        assert qr_image.size[0] > 0
        assert qr_image.size[1] > 0

def test_different_colors():
    """Test QR code generation with different colors."""
    data = "https://example.com"
    
    # Test different color combinations
    color_combinations = [
        ("black", "white"),
        ("blue", "yellow"),
        ("red", "green"),
        ("#000000", "#FFFFFF"),
        ("rgb(0,0,0)", "rgb(255,255,255)")
    ]
    
    for fill_color, back_color in color_combinations:
        qr = QRGenerator(fill_color=fill_color, back_color=back_color)
        qr_image = qr.generate(data)
        
        assert isinstance(qr_image, Image.Image)
        assert qr_image.mode == 'RGB'
        assert qr_image.size[0] > 0
        assert qr_image.size[1] > 0

def test_different_box_sizes():
    """Test QR code generation with different box sizes."""
    data = "https://example.com"
    
    # Test different box sizes
    for box_size in [5, 10, 15, 20]:
        qr = QRGenerator(box_size=box_size)
        qr_image = qr.generate(data)
        
        assert isinstance(qr_image, Image.Image)
        assert qr_image.mode == 'RGB'
        assert qr_image.size[0] > 0
        assert qr_image.size[1] > 0

def test_different_borders():
    """Test QR code generation with different border sizes."""
    data = "https://example.com"
    
    # Test different border sizes
    for border in [0, 2, 4, 8]:
        qr = QRGenerator(border=border)
        qr_image = qr.generate(data)
        
        assert isinstance(qr_image, Image.Image)
        assert qr_image.mode == 'RGB'
        assert qr_image.size[0] > 0
        assert qr_image.size[1] > 0 