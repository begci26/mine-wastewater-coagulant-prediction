from flask import Blueprint, render_template

# Initialize blueprint
about_bp = Blueprint('about', __name__)

AUTHOR_INFO = {
    "name": "Meidyakama Arsya",
    "nim": "2411600279",
}


@about_bp.route('/')
def index():
    """Renders the about page containing research context and details."""
    return render_template('about.html', author=AUTHOR_INFO)
