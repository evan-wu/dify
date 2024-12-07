"""merge heads

Revision ID: ea003902b54e
Revises: d9605fd15e8c, 01d6889832f7
Create Date: 2024-12-07 11:10:18.283795

"""
from alembic import op
import models as models
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ea003902b54e'
down_revision = ('d9605fd15e8c', '01d6889832f7')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
